"""Write the sample harbor brief used to try the agent."""

from __future__ import annotations

from rag_agent.config import SAMPLE_PDF
from rag_agent.pdf_writer import write_text_pdf

PAGES: list[list[str]] = [
    [
        "Northwind Harbor Operations Brief",
        "Call date: 18 March 2026. Prepared for the night-shift docking team.",
        (
            "The vessel Kestrel arrives at Northwind Harbor on 18 March 2026. "
            "The docking window is 04:30 to 06:15 local time. "
            "The assigned berth is Pier 7, berth C."
        ),
        (
            "The harbor master on duty is Elena Voss. "
            "Reach the harbor master on radio channel 14."
        ),
    ],
    [
        "Cargo and restrictions",
        (
            "The cargo is 1240 crates of cold-chain salmon. "
            "Crates must stay at or below 2 degrees Celsius. "
            "The invoice code for this call is NH-4481."
        ),
        (
            "Night crane operations stop when sustained wind exceeds 25 knots. "
            "The backup berth, if Pier 7 is occupied, is Pier 3, berth A."
        ),
        (
            "Tug assistance is required because the Kestrel has a draft of 9.4 meters."
        ),
    ],
    [
        "Boarding and contacts",
        (
            "Pilot boarding is at buoy NH-2. The agent is Marlow and Finch. "
            "The agent phone number is 555-0148."
        ),
        (
            "Do not discharge ballast inside the inner basin. "
            "Fuel barge access is on the south side of Pier 7 and opens after 07:00."
        ),
    ],
]


def write_sample(path=SAMPLE_PDF) -> None:
    write_text_pdf(path, PAGES)


def main() -> None:
    write_sample()
    print(SAMPLE_PDF)


if __name__ == "__main__":
    main()
