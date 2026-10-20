"""Seed sample PDF script for demo preparation (Phase 8 Stage 15).

Generates `data/samples/AUTOSAR_HLD_Sample_BrakingSystem.pdf` for live demo execution.
"""

from pathlib import Path
import pymupdf


def generate_sample_pdf(output_path: Path) -> Path:
    """Create authentic sample AUTOSAR HLD PDF specification."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()

    # Page 1: 1.0 Architecture Overview
    p1 = doc.new_page(width=595, height=842)  # A4
    p1.insert_text(
        (50, 70),
        "AUTOSAR HIGH-LEVEL DESIGN SPECIFICATION\n"
        "Document: AUTOSAR_HLD_Sample_BrakingSystem\n"
        "Classification: Automotive Technical Specification\n\n"
        "1.0 Architecture Overview\n"
        "The SWC_BrakingController is responsible for vehicle deceleration arbitration.\n"
        "It interacts with chassis sensors and powertrain actuators.\n"
        "The subsystem manages anti-lock braking modulation, hydraulic pressure blending,\n"
        "and regenerative braking command coordination.",
        fontsize=11,
    )

    # Page 2: 2.0 Port Specifications Table
    p2 = doc.new_page(width=595, height=842)  # A4
    p2.insert_text(
        (50, 70),
        "2.0 Port Specifications for SWC_BrakingController\n\n"
        "The following interface ports are defined for the component:\n\n"
        "| Port Name | Direction | Interface | Description |\n"
        "|---|---|---|---|\n"
        "| P_BrakePressure | Provided | If_BrakePressure | Current hydraulic pressure |\n"
        "| R_PedalPosition | Required | If_PedalPosition | Driver brake demand |\n\n"
        "Notes: P_BrakePressure broadcasts sensor readings at 10ms periodic rate.\n"
        "R_PedalPosition receives redundant sensor readings for ISO 26262 ASIL-D safety integrity.",
        fontsize=11,
    )

    doc.save(str(output_path))
    doc.close()
    return output_path


if __name__ == "__main__":
    sample_dir = Path(__file__).parent.parent / "data" / "samples"
    dest = sample_dir / "AUTOSAR_HLD_Sample_BrakingSystem.pdf"
    generate_sample_pdf(dest)
    print(f"Sample PDF generated at: {dest}")
