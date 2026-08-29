"""Standalone Restaurant Booking Assistant.

Run with:
    uvicorn apps.restaurant_booking_assistant:app --reload

Set GROQ_API_KEY (or the legacy GROK_API_CLOUD_KEY) in a .env file first.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

load_dotenv()
if not os.getenv("GROQ_API_KEY") and os.getenv("GROK_API_CLOUD_KEY"):
    os.environ["GROQ_API_KEY"] = os.environ["GROK_API_CLOUD_KEY"]


class BookingDetails(BaseModel):
    """The JSON contract extracted from each conversation turn."""

    name: str | None = Field(None, description="Guest's full name")
    reservation_date: date | None = Field(None, description="Reservation date in YYYY-MM-DD")
    reservation_time: time | None = Field(None, description="Reservation time in HH:MM, 24-hour time")
    guests: int | None = Field(None, description="Number of guests")
    preferences: str | None = Field(None, description="Dietary, cuisine, or seating preferences")
    phone: str | None = Field(None, description="Guest contact phone number")

    @field_validator("name", "preferences", "phone", mode="before")
    @classmethod
    def strip_strings(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


def validate_booking(booking: BookingDetails, today: date | None = None) -> list[str]:
    """Apply restaurant rules after Pydantic validates the JSON shape and types."""
    today = today or date.today()
    errors: list[str] = []
    if booking.reservation_date and booking.reservation_date < today:
        errors.append("The reservation date cannot be in the past.")
    if booking.reservation_time and not time(11, 30) <= booking.reservation_time <= time(22, 0):
        errors.append("Reservations are available from 11:30 to 22:00.")
    if booking.guests is not None and not 1 <= booking.guests <= 15:
        errors.append("Online bookings must be for between 1 and 15 guests.")
    return errors


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    booking: BookingDetails = Field(default_factory=BookingDetails)
    history: list[tuple[Literal["user", "assistant"], str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    booking: BookingDetails
    validation_errors: list[str]
    raw_json: str | None = None


class LegacyChatRequest(BaseModel):
    """Payload used by the existing browser UI in apps/static/index.html."""

    user_message: str = Field(min_length=1, max_length=2_000)
    chat_history: list[tuple[Literal["user", "assistant"], str]] = Field(default_factory=list)
    booking_state: dict[str, Any] = Field(default_factory=dict)


class RestaurantBookingAssistant:
    """LLM extraction + Pydantic parsing + domain validation in one pipeline."""

    required_fields = ("name", "reservation_date", "reservation_time", "guests")

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured. Add it to .env before starting the server.")
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=BookingDetails)
        self.extraction_prompt = PromptTemplate(
            template="""Extract restaurant booking details from the conversation as JSON.
Today is {today}. Resolve relative dates using this date. Preserve values in the current booking unless the user changes them. Never invent a value; use null when unknown. Times must be 24-hour HH:MM.

Current booking:
{booking}

Conversation:
{history}

{format_instructions}
Return only the JSON object.""",
            input_variables=["today", "booking", "history"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def process(self, request: ChatRequest) -> ChatResponse:
        history = request.history + [("user", request.message)]
        history = history[-10:]  # only keep the last 10 messages for context
        history_text = "\n".join(f"{role.title()}: {message}" for role, message in history)
        print("history_text----", history_text)
        prompt = self.extraction_prompt.format(
            today=date.today().isoformat(),
            booking=request.booking.model_dump_json(),
            history=history_text,
        )
        print("prompt----", prompt)
        raw_json = self.llm.invoke(prompt).content.strip()
        print("raw_json----", raw_json)
        try:
            extracted = self.parser.parse(raw_json)
            print("extracted booking details----", extracted)
        except Exception as exc:
            return ChatResponse(
                reply="I couldn't read the booking details reliably. Please share the date, time, and party size again.",
                booking=request.booking,
                validation_errors=[f"JSON response validation failed: {exc}"],
                raw_json=raw_json,
            )

        errors = validate_booking(extracted)
        if errors:
            return ChatResponse(
                reply=" ".join(errors) + " Please provide a suitable alternative.",
                booking=request.booking,
                validation_errors=errors,
                raw_json=raw_json,
            )

        missing = [field.replace("_", " ") for field in self.required_fields if getattr(extracted, field) is None]
        if missing:
            return ChatResponse(
                reply=f"Thanks. To complete your reservation, please provide your {', '.join(missing)}.",
                booking=extracted,
                validation_errors=[],
                raw_json=raw_json,
            )
        confirmation_messages = {
            "confirm", "confirmed", "yes", "yes please", "please confirm",
            "sure", "that's correct", "that is correct",
        }
        normalized_message = request.message.casefold().strip(" \t\r\n.,!?*_~")
        if normalized_message in confirmation_messages:
            return ChatResponse(
                reply=(f"Your reservation is confirmed for {extracted.guests} under {extracted.name} on "
                       f"{extracted.reservation_date:%B %d, %Y} at {extracted.reservation_time:%H:%M}. "
                       "Confirmation Code: BI-99217. We look forward to welcoming you!"),
                booking=extracted,
                validation_errors=[],
                raw_json=raw_json,
            )

        return ChatResponse(
            reply=(f"I have a table for {extracted.guests} under {extracted.name} on "
                   f"{extracted.reservation_date:%B %d, %Y} at {extracted.reservation_time:%H:%M}. "
                   "Reply 'confirm' to finalize it."),
            booking=extracted,
            validation_errors=[],
            raw_json=raw_json,
        )


app = FastAPI(title="Restaurant Booking Assistant")
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/api/booking-chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return RestaurantBookingAssistant().process(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to process this booking request.") from exc


@app.post("/api/chat")
def legacy_chat(request: LegacyChatRequest) -> dict[str, Any]:
    """Compatibility endpoint for the full visual UI shipped with this project."""
    state = request.booking_state
    print("state----", state)
    print("hyyyyyy--------")
    booking = BookingDetails(
        name=state.get("name"),
        reservation_date=state.get("date"),
        reservation_time=state.get("time"),
        guests=state.get("guests"),
        preferences=state.get("cuisine_pref"),
        phone=state.get("contact_phone"),
    )
    print("booking details----", booking)
    print("(BookingDetails--------", BookingDetails.model_json_schema())
    result = chat(ChatRequest(message=request.user_message, booking=booking, history=request.chat_history))
    extracted = result.booking
    return {
        "agent_reply": result.reply,
        "booking_state": {
            "name": extracted.name,
            "date": extracted.reservation_date.isoformat() if extracted.reservation_date else None,
            "time": extracted.reservation_time.strftime("%H:%M") if extracted.reservation_time else None,
            "guests": extracted.guests,
            "cuisine_pref": extracted.preferences,
            "contact_phone": extracted.phone,
        },
        "chat_history": request.chat_history + [("user", request.user_message), ("assistant", result.reply)],
        "validation_errors": result.validation_errors,
    }


@app.get("/", response_class=HTMLResponse)
def visual_home() -> FileResponse:
    """Serve the full Bella Italia UI in apps/static/index.html."""
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.restaurant_booking_assistant:app", host="0.0.0.0", port=8000, reload=True)
