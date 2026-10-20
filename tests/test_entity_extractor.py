"""Unit tests for Phase 6: AUTOSAR entity extraction MVP.

Tests verify:
1. Pydantic schema validation for SoftwareComponent, Port, and Interface models.
2. Rule-based extraction from AUTOSAR Markdown port tables.
3. Rule-based extraction from structured text and naming patterns.
4. LLM-assisted extraction using MockLLMProvider (no external API calls).
5. Extraction across multiple document Chunks.
6. Entity deduplication within a document.
7. Persistence and retrieval via DatabaseManager (SQLite DAO).
"""

import json
from pathlib import Path
import pytest
from core.chunking.models import Chunk
from core.extraction.entity_extractor import EntityExtractor
from core.extraction.models import (
    EntityExtractionResult,
    Interface,
    Port,
    SoftwareComponent,
)
from core.llm.mock_provider import MockLLMProvider
from storage.database import DatabaseManager


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    """Fixture providing a clean SQLite test database."""
    db_file = tmp_path / "test_autosar.db"
    return DatabaseManager(db_path=db_file)


class TestEntityModels:
    """Tests for Pydantic domain models."""

    def test_software_component_validation(self) -> None:
        swc = SoftwareComponent(
            name="  SWC_BatteryManagement  ",
            component_type="ApplicationSwComponentType",
            description="Manages high voltage battery state",
            document_id="doc_hld_01",
            page_number=3,
            section="3.1 Battery Subsystem",
        )
        assert swc.name == "SWC_BatteryManagement"
        assert swc.page_number == 3
        assert swc.section == "3.1 Battery Subsystem"

    def test_software_component_empty_name_fails(self) -> None:
        with pytest.raises(ValueError):
            SoftwareComponent(
                name="   ",
                document_id="doc_01",
                page_number=1,
            )

    def test_port_normalization(self) -> None:
        p1 = Port(
            name="P_BmsStatus",
            direction="provided",
            swc_parent="SWC_BMS",
            document_id="doc_01",
            page_number=2,
        )
        assert p1.direction == "Provided"

        p2 = Port(
            name="R_VehicleSpeed",
            direction="rport",
            swc_parent="SWC_BMS",
            document_id="doc_01",
            page_number=2,
        )
        assert p2.direction == "Required"

        p3 = Port(
            name="PR_CalibrationData",
            direction="PRPort",
            document_id="doc_01",
            page_number=2,
        )
        assert p3.direction == "ProvidedRequired"

    def test_interface_normalization(self) -> None:
        i1 = Interface(
            name="If_BatteryStatus",
            interface_type="sr",
            document_id="doc_01",
            page_number=4,
        )
        assert i1.interface_type == "SenderReceiverInterface"

        i2 = Interface(
            name="If_DiagnosticService_CS",
            interface_type="ClientServer",
            document_id="doc_01",
            page_number=4,
        )
        assert i2.interface_type == "ClientServerInterface"

        i3 = Interface(
            name="If_ThermalParam",
            interface_type="Parameter",
            document_id="doc_01",
            page_number=4,
        )
        assert i3.interface_type == "ParameterInterface"


class TestRuleBasedExtraction:
    """Tests for deterministic table and pattern extraction."""

    def test_extract_from_markdown_table(self) -> None:
        markdown_excerpt = """
### Section 4.2: Port Interfaces for SWC_BmsController

The component defines the following port specifications:

| Port Name | Direction | Interface | Description |
|---|---|---|---|
| P_BmsCellVoltage | Provided | If_CellVoltage | Broadcasts cell voltages |
| R_AmbientTemp | Required | If_AmbientTemp | Receives cabin/exterior temperature |
| PR_BalancingControl | ProvidedRequired | If_Balancing | Manages active cell balancing |
        """

        extractor = EntityExtractor()
        result = extractor.extract_from_text(
            text=markdown_excerpt,
            document_id="HLD_BMS_v1",
            page_number=12,
            section="4.2 Port Interfaces",
        )

        assert isinstance(result, EntityExtractionResult)
        assert result.document_id == "HLD_BMS_v1"
        assert result.total_entities >= 6

        # Check detected SWC
        swc_names = [s.name for s in result.software_components]
        assert "SWC_BmsController" in swc_names

        # Check ports
        port_names = [p.name for p in result.ports]
        assert "P_BmsCellVoltage" in port_names
        assert "R_AmbientTemp" in port_names
        assert "PR_BalancingControl" in port_names

        p_volt = next(p for p in result.ports if p.name == "P_BmsCellVoltage")
        assert p_volt.direction == "Provided"
        assert p_volt.interface_name == "If_CellVoltage"
        assert p_volt.swc_parent == "SWC_BmsController"
        assert p_volt.page_number == 12

        # Check interfaces created from table
        iface_names = [i.name for i in result.interfaces]
        assert "If_CellVoltage" in iface_names
        assert "If_AmbientTemp" in iface_names

    def test_extract_from_narrative_patterns(self) -> None:
        narrative_text = """
The SWC_MotorDrive component interfaces with the chassis domain.
It transmits through P_MotorTorque and listens on R_ThrottlePosition.
Communication is governed by Sender-Receiver Interface: If_TorqueCommand.
        """

        extractor = EntityExtractor()
        result = extractor.extract_from_text(
            text=narrative_text,
            document_id="HLD_Motor_02",
            page_number=5,
            section="2.1 Motor Overview",
        )

        assert any(s.name == "SWC_MotorDrive" for s in result.software_components)
        assert any(p.name == "P_MotorTorque" and p.direction == "Provided" for p in result.ports)
        assert any(p.name == "R_ThrottlePosition" and p.direction == "Required" for p in result.ports)
        assert any(i.name == "If_TorqueCommand" for i in result.interfaces)


class TestLLMAssistedExtraction:
    """Tests for LLM-assisted extraction using MockLLMProvider."""

    def test_llm_assisted_extraction_success(self) -> None:
        llm_response_json = json.dumps({
            "software_components": [
                {
                    "name": "SWC_ChargingSupervision",
                    "component_type": "ApplicationSwComponentType",
                    "description": "Monitors EV charging states",
                }
            ],
            "ports": [
                {
                    "name": "P_ChargeState",
                    "direction": "Provided",
                    "swc_parent": "SWC_ChargingSupervision",
                    "interface_name": "If_ChargeState",
                    "description": "Broadcasts current charging phase",
                }
            ],
            "interfaces": [
                {
                    "name": "If_ChargeState",
                    "interface_type": "SenderReceiverInterface",
                    "description": "Payload with StateOfCharge enum",
                }
            ],
        })

        mock_llm = MockLLMProvider(canned_response=llm_response_json)
        extractor = EntityExtractor(llm_provider=mock_llm)

        unstructured_text = "Charging supervision logic is encapsulated here."
        result = extractor.extract_from_text(
            text=unstructured_text,
            document_id="HLD_Charge_03",
            page_number=8,
            section="5.0 Charging",
            use_llm=True,
        )

        assert len(result.software_components) == 1
        assert result.software_components[0].name == "SWC_ChargingSupervision"
        assert len(result.ports) == 1
        assert result.ports[0].name == "P_ChargeState"
        assert len(result.interfaces) == 1
        assert result.interfaces[0].name == "If_ChargeState"


class TestChunkExtractionAndDeduplication:
    """Tests for chunk processing and deduplication."""

    def test_extract_from_chunks_with_duplicates(self) -> None:
        from core.chunking.models import ChunkMetadata

        chunk1 = Chunk(
            chunk_id="chunk_01",
            text="SWC_SteeringControl uses P_SteeringAngle to broadcast position.",
            document_id="HLD_Steering",
            page_number=1,
            section_title="1. Steering",
            start_char=0,
            end_char=60,
            metadata=ChunkMetadata(
                document_id="HLD_Steering",
                page_number=1,
                section_title="1. Steering",
                start_char=0,
                end_char=60,
                chunk_index=0,
            ),
        )
        chunk2 = Chunk(
            chunk_id="chunk_02",
            text="SWC_SteeringControl also defines P_SteeringAngle with additional remarks.",
            document_id="HLD_Steering",
            page_number=2,
            section_title="1.1 Steering Details",
            start_char=0,
            end_char=70,
            metadata=ChunkMetadata(
                document_id="HLD_Steering",
                page_number=2,
                section_title="1.1 Steering Details",
                start_char=0,
                end_char=70,
                chunk_index=1,
            ),
        )

        extractor = EntityExtractor()
        result = extractor.extract_from_chunks([chunk1, chunk2], document_id="HLD_Steering")

        # SWC and Port should be deduplicated
        assert len(result.software_components) == 1
        assert result.software_components[0].name == "SWC_SteeringControl"
        assert len(result.ports) == 1
        assert result.ports[0].name == "P_SteeringAngle"

    def test_to_dict_format(self) -> None:
        swc = SoftwareComponent(
            name="SWC_BMS",
            document_id="doc_1",
            page_number=1,
        )
        res = EntityExtractionResult(
            document_id="doc_1",
            software_components=[swc],
        )
        data = res.to_dict()
        assert data["document_id"] == "doc_1"
        assert data["total_entities"] == 1
        assert len(data["software_components"]) == 1
        assert data["software_components"][0]["name"] == "SWC_BMS"


class TestDatabasePersistence:
    """Tests for persisting and retrieving extracted entities in SQLite."""

    def test_persistence_and_retrieval(self, temp_db: DatabaseManager) -> None:
        text = """
SWC_PowertrainSupervisor defines:
| Port Name | Direction | Interface | Description |
|---|---|---|---|
| P_PowerState | Provided | If_PowerState | Main power state |
| R_GearPosition | Required | If_GearPosition | Current gear |
        """

        extractor = EntityExtractor(db_manager=temp_db)
        result = extractor.extract_from_text(
            text=text,
            document_id="HLD_PT_01",
            page_number=7,
            section="Section 7",
        )

        assert result.total_entities >= 5

        # Query database directly
        db_swcs = temp_db.get_software_components("HLD_PT_01")
        assert len(db_swcs) == 1
        assert db_swcs[0]["name"] == "SWC_PowertrainSupervisor"
        assert db_swcs[0]["page_number"] == 7

        db_ports = temp_db.get_ports("HLD_PT_01")
        assert len(db_ports) == 2
        port_names = [p["name"] for p in db_ports]
        assert "P_PowerState" in port_names
        assert "R_GearPosition" in port_names

        db_ifaces = temp_db.get_interfaces("HLD_PT_01")
        assert len(db_ifaces) >= 2
        iface_names = [i["name"] for i in db_ifaces]
        assert "If_PowerState" in iface_names

    def test_database_deduplication_on_reinsert(self, temp_db: DatabaseManager) -> None:
        swc1 = SoftwareComponent(
            name="SWC_BMS",
            component_type="ApplicationSwComponentType",
            description="Initial version",
            document_id="doc_01",
            page_number=1,
        )
        temp_db.save_software_components([swc1])
        assert len(temp_db.get_software_components("doc_01")) == 1

        # Reinsert same component with updated description
        swc2 = SoftwareComponent(
            name="SWC_BMS",
            component_type="ApplicationSwComponentType",
            description="Updated description",
            document_id="doc_01",
            page_number=2,
        )
        temp_db.save_software_components([swc2])
        rows = temp_db.get_software_components("doc_01")
        assert len(rows) == 1
        assert rows[0]["description"] == "Updated description"
        assert rows[0]["page_number"] == 2
