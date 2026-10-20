"""Pydantic data models for AUTOSAR domain entities (Phase 6).

Covers:
- Software Components (SWC)
- Ports (PPort, RPort, PRPort)
- Interfaces (SenderReceiver, ClientServer, Parameter)
- Extraction Results container
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class SoftwareComponent(BaseModel):
    """Represents an AUTOSAR Software Component (SWC)."""

    name: str = Field(..., description="Name of the software component, e.g., 'SWC_BatteryManagement'")
    component_type: str = Field(
        default="ApplicationSwComponentType",
        description="Type of SWC (e.g. ApplicationSwComponentType, SensorActuatorSwComponentType, ServiceSwComponentType)",
    )
    description: Optional[str] = Field(None, description="Functional summary or description")
    document_id: str = Field(..., description="Identifier of the source document")
    page_number: int = Field(..., ge=1, description="Page number where the entity was discovered")
    section: Optional[str] = Field(None, description="Section heading context")

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        val = v.strip()
        if not val:
            raise ValueError("SoftwareComponent name cannot be empty")
        return val


class Port(BaseModel):
    """Represents an AUTOSAR Port (PPort, RPort, PRPort)."""

    name: str = Field(..., description="Name of the port, e.g., 'P_BatteryStatus'")
    direction: str = Field(..., description="Direction/type of port: 'Provided' (PPort), 'Required' (RPort), or 'ProvidedRequired' (PRPort)")
    swc_parent: Optional[str] = Field(None, description="Name of the enclosing Software Component")
    interface_name: Optional[str] = Field(None, description="Name of the connected interface")
    description: Optional[str] = Field(None, description="Description or role of the port")
    document_id: str = Field(..., description="Identifier of the source document")
    page_number: int = Field(..., ge=1, description="Page number where the entity was discovered")
    section: Optional[str] = Field(None, description="Section heading context")

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        val = v.strip()
        if not val:
            raise ValueError("Port name cannot be empty")
        return val

    @field_validator("direction")
    @classmethod
    def normalize_direction(cls, v: str) -> str:
        val = v.strip()
        lower = val.lower()
        if "req" in lower or lower in ("r", "rport", "in", "input"):
            return "Required"
        if "prov" in lower or lower in ("p", "pport", "out", "output"):
            return "Provided"
        if "both" in lower or "pr" in lower or "prport" in lower:
            return "ProvidedRequired"
        return val.capitalize() or "Provided"


class Interface(BaseModel):
    """Represents an AUTOSAR Interface (SenderReceiver, ClientServer, Parameter)."""

    name: str = Field(..., description="Name of the interface, e.g., 'If_BatteryStatus'")
    interface_type: str = Field(
        default="SenderReceiverInterface",
        description="Type of interface: SenderReceiverInterface, ClientServerInterface, or ParameterInterface",
    )
    description: Optional[str] = Field(None, description="Summary or description of the interface payload/operations")
    document_id: str = Field(..., description="Identifier of the source document")
    page_number: int = Field(..., ge=1, description="Page number where the entity was discovered")
    section: Optional[str] = Field(None, description="Section heading context")

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        val = v.strip()
        if not val:
            raise ValueError("Interface name cannot be empty")
        return val

    @field_validator("interface_type")
    @classmethod
    def normalize_interface_type(cls, v: str) -> str:
        val = v.strip()
        lower = val.lower()
        if "client" in lower or "server" in lower or "cs" in lower:
            return "ClientServerInterface"
        if "param" in lower:
            return "ParameterInterface"
        if "sender" in lower or "receiver" in lower or "sr" in lower:
            return "SenderReceiverInterface"
        return val or "SenderReceiverInterface"


class EntityExtractionResult(BaseModel):
    """Container for validated extracted entities for a document or batch."""

    document_id: str = Field(..., description="Identifier of the source document")
    software_components: list[SoftwareComponent] = Field(default_factory=list)
    ports: list[Port] = Field(default_factory=list)
    interfaces: list[Interface] = Field(default_factory=list)

    @property
    def total_entities(self) -> int:
        """Total count of all extracted entities."""
        return len(self.software_components) + len(self.ports) + len(self.interfaces)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary representation."""
        return {
            "document_id": self.document_id,
            "total_entities": self.total_entities,
            "software_components": [c.model_dump() for c in self.software_components],
            "ports": [p.model_dump() for p in self.ports],
            "interfaces": [i.model_dump() for i in self.interfaces],
        }
