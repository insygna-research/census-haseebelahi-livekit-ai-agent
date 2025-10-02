import logging
import json
import csv
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


async def extract_customer_metadata(ctx: JobContext, max_retries: int = 5) -> Dict[str, Any]:
    """Extract customer metadata from LiveKit room and participant attributes with retries"""
    customer_data = {}

    # Try multiple times to get metadata as participants may not be immediately available
    for attempt in range(max_retries):
        try:
            logger.info(f"Finding customer metadata... (attempt {attempt + 1}/{max_retries})")
            logger.info(f"Room metadata: {ctx.room.metadata}")
            logger.info(f"Remote participants: {list(ctx.room.remote_participants.keys())}")

            # Get room metadata (call context)
            if ctx.room.metadata:
                room_metadata = json.loads(ctx.room.metadata)
                customer_data.update(room_metadata)
                logger.info(f"Extracted room metadata: {room_metadata}")
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse room metadata: {e}")

        # Find SIP participant and extract customer attributes
        try:
            # Access participants through the room's remote participants
            for participant in ctx.room.remote_participants.values():
                if participant.identity.startswith('caller-'):
                    try:
                        # Extract customer attributes from SIP participant
                        attributes = participant.attributes
                        customer_data.update(attributes)
                        logger.info(f"Extracted customer attributes: {attributes}")

                        # If we found customer data, we can stop retrying
                        if customer_data.get('customer_phone'):
                            logger.info("Successfully extracted customer metadata!")
                            return customer_data

                    except Exception as e:
                        logger.warning(f"Failed to extract participant attributes: {e}")
        except Exception as e:
            logger.warning(f"Failed to access remote participants: {e}")

        # Wait before retrying (except on last attempt)
        if attempt < max_retries - 1:
            logger.info(f"Retrying metadata extraction in 0.5 seconds...")
            await asyncio.sleep(0.5)

    # If no participants found after all retries, log a warning but continue with graceful fallback
    if not customer_data.get('customer_phone'):
        logger.warning("No customer metadata found after retries - using default greeting")

    return customer_data


def generate_personalized_instructions(customer_data: Dict[str, Any]) -> str:
    """Generate personalized instructions for the AI agent based on customer data"""

    # Base instructions for car dealership maintenance calls
    base_instructions = """You are a friendly and professional representative from a car dealership and workshop calling to help customers schedule their regular vehicle maintenance.

Always start by greeting the customer with "Hi Sir!" and introduce yourself as calling from the car service center. Never use the customer's name.

Your goal is to:
1. Engage in brief, friendly chit-chat using car details and service history
2. Mention their specific car (make and model) to show you have their records
3. Remind them about their maintenance schedule based on mileage
4. Help them book an appointment for their regular service
5. Be helpful, professional, and not pushy

Keep responses concise and conversational. Avoid complex formatting, emojis, or symbols."""

    # Add personalization based on customer data
    personalization = "\n\nCustomer Context:\n"

    # Car details - This is the main focus for the conversation
    car_make = customer_data.get('custom_car_make', '')
    car_model = customer_data.get('custom_car_model', '')
    if car_make and car_model:
        personalization += f"- Customer's vehicle: {car_make} {car_model}\n"
        personalization += f"- Mention their {car_make} {car_model} early in the conversation to show you have their records\n"
    elif car_make:
        personalization += f"- Customer's vehicle make: {car_make}\n"

    # Mileage information for maintenance scheduling
    last_mileage = customer_data.get('custom_car_last_milleage', '')
    expected_mileage = customer_data.get('customer_car_expected_milleage', '')
    if last_mileage and expected_mileage:
        personalization += f"- Last recorded mileage: {last_mileage} miles\n"
        personalization += f"- Expected current mileage: {expected_mileage} miles\n"
        personalization += f"- Use this mileage info to discuss maintenance needs\n"
    elif last_mileage:
        personalization += f"- Last recorded mileage: {last_mileage} miles\n"

    # Customer type personalization
    customer_type = customer_data.get('customer_type', 'standard')
    if customer_type == 'premium':
        personalization += "- This is a premium customer - be extra attentive and offer premium services\n"
    elif customer_type == 'new':
        personalization += "- This is a new customer - be welcoming and explain your services clearly\n"

    # Special notes for conversation starters
    special_notes = customer_data.get('special_notes', '')
    if special_notes:
        personalization += f"- Use this for chit-chat: {special_notes}\n"

    # Purchase/service history
    purchase_history = customer_data.get('purchase_history', '')
    if purchase_history:
        personalization += f"- Last service was: {purchase_history}\n"
        personalization += "- Use this to remind them it's time for their next maintenance\n"

    # Account status
    account_status = customer_data.get('account_status', '')
    if account_status == 'inactive':
        personalization += "- Customer account is inactive - be extra welcoming to re-engage them\n"

    return base_instructions + personalization


class Assistant(Agent):
    def __init__(self, instructions: str = None, customer_data: Dict[str, Any] = None) -> None:
        # Use provided instructions or fall back to default
        default_instructions = """You are a helpful voice AI assistant.
        You eagerly assist users with their questions by providing information from your extensive knowledge.
        Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
        You are curious, friendly, and have a sense of humor."""

        super().__init__(
            instructions=instructions or default_instructions,
        )
        self.customer_data = customer_data or {}

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active

    def _get_appointments_csv_path(self) -> str:
        """Get the path to the appointments CSV file"""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "appointments.csv")

    def _ensure_appointments_csv(self) -> None:
        """Ensure appointments CSV file exists with proper headers"""
        csv_path = self._get_appointments_csv_path()
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['customer_phone', 'vehicle_info', 'date', 'time', 'service_type', 'booked_at'])

    def _parse_day_and_time(self, date_str: str, time_str: str) -> tuple[str, str]:
        """Parse user input for date and time into standardized format"""
        # Simple parsing - in production, you'd want more robust date parsing
        date_str = date_str.lower().strip()
        time_str = time_str.lower().strip()

        # Map common day references
        today = datetime.now()
        day_mapping = {
            'today': today.strftime('%Y-%m-%d'),
            'tomorrow': (today + timedelta(days=1)).strftime('%Y-%m-%d'),
            'monday': self._get_next_weekday(0),
            'tuesday': self._get_next_weekday(1),
            'wednesday': self._get_next_weekday(2),
            'thursday': self._get_next_weekday(3),
            'friday': self._get_next_weekday(4),
            'saturday': self._get_next_weekday(5),
        }

        parsed_date = day_mapping.get(date_str, date_str)

        # Map common time references
        time_mapping = {
            'morning': '10:00',
            'afternoon': '14:00',
            'evening': '17:00',
        }

        parsed_time = time_mapping.get(time_str, time_str)

        return parsed_date, parsed_time

    def _get_next_weekday(self, weekday: int) -> str:
        """Get the date of the next occurrence of a weekday (0=Monday, 6=Sunday)"""
        today = datetime.now()
        days_ahead = weekday - today.weekday()
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return (today + timedelta(days_ahead)).strftime('%Y-%m-%d')

    @function_tool
    async def check_available_times(self, context: RunContext, date: str):
        """Check available appointment times for a specific date.

        Args:
            date: The date to check availability for (e.g. "tomorrow", "Monday", "January 15th")
        """
        logger.info(f"Checking availability for {date}")

        self._ensure_appointments_csv()
        parsed_date, _ = self._parse_day_and_time(date, "")

        # Read existing bookings
        csv_path = self._get_appointments_csv_path()
        booked_times = set()

        with open(csv_path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['date'] == parsed_date:
                    booked_times.add(row['time'])

        # Generate available times (10am-6pm, every hour)
        all_times = [f"{hour:02d}:00" for hour in range(10, 19)]  # 10:00 to 18:00
        available_times = [time for time in all_times if time not in booked_times]

        if available_times:
            return f"For {date} ({parsed_date}), we have these times available: {', '.join(available_times)}. Which time works best for you?"
        else:
            return f"I'm sorry, but {date} ({parsed_date}) is fully booked. Would you like to check another day?"

    @function_tool
    async def book_maintenance_appointment(
        self,
        context: RunContext,
        preferred_date: str,
        preferred_time: str,
        service_type: str = "regular_maintenance"
    ):
        """Use this tool to book a maintenance appointment for the customer.

        Args:
            preferred_date: The customer's preferred date (e.g. "next Monday", "January 15th")
            preferred_time: The customer's preferred time (e.g. "morning", "2 PM", "afternoon")
            service_type: Type of service needed (default: "regular_maintenance")
        """
        logger.info(f"Booking appointment: {preferred_date} at {preferred_time} for {service_type}")

        self._ensure_appointments_csv()
        parsed_date, parsed_time = self._parse_day_and_time(preferred_date, preferred_time)

        # Check if slot is available
        csv_path = self._get_appointments_csv_path()
        with open(csv_path, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['date'] == parsed_date and row['time'] == parsed_time:
                    return f"I'm sorry, but {preferred_date} at {preferred_time} is already booked. Let me check other available times for you."

        # Get customer info from stored customer data
        customer_phone = self.customer_data.get('customer_phone', 'unknown')
        car_make = self.customer_data.get('custom_car_make', '')
        car_model = self.customer_data.get('custom_car_model', '')

        # Create a vehicle identifier instead of using customer name
        vehicle_info = f"{car_make} {car_model}".strip() if car_make or car_model else "Vehicle"
        if not vehicle_info or vehicle_info == " ":
            vehicle_info = "Customer Vehicle"

        # Book the appointment
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                customer_phone,
                vehicle_info,  # Using vehicle info instead of customer name
                parsed_date,
                parsed_time,
                service_type,
                datetime.now().isoformat()
            ])

        return f"Perfect! I've scheduled your {service_type} appointment for {preferred_date} ({parsed_date}) at {parsed_time}. You'll receive a confirmation call one day before your appointment. Is there anything else I can help you with today?"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Only handle dealership rooms
    logger.info(f"Agent received job for room: {ctx.room.name}")
    if not ctx.room.name.startswith("dealership-test-"):
        logger.info(f"Skipping non-dealership room: {ctx.room.name}")
        return

    logger.info(f"Car dealership agent joining room: {ctx.room.name}")

    # Connect to the room first
    await ctx.connect()

    # Wait a moment for participants and metadata to be fully populated
    await asyncio.sleep(1.0)

    # Now extract customer metadata after connection is established
    logger.info("Connection established, extracting customer metadata...")
    customer_data = await extract_customer_metadata(ctx)
    logger.info(f"Customer data extracted: {customer_data}")

    # Generate personalized instructions
    personalized_instructions = generate_personalized_instructions(customer_data)
    logger.info("Generated personalized instructions for customer")

    # Update logging context with customer data
    ctx.log_context_fields.update({
        "customer_phone": customer_data.get("customer_phone", "unknown"),
        "customer_type": customer_data.get("customer_type", "standard"),
    })

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM.with_deepseek(
        model="deepseek-chat"), # this is DeepSeek-V3,
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="multi"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session with personalized assistant (now with customer data)
    await session.start(
        agent=Assistant(instructions=personalized_instructions, customer_data=customer_data),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
