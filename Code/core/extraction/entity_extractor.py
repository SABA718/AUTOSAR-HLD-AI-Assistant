"""AUTOSAR entity extraction module (Phase 6 MVP).

Extracts and validates:
1. Software Components (SWC)
2. Ports (PPort, RPort, PRPort)
3. Interfaces (SenderReceiver, ClientServer, Parameter)

Features:
- Deterministic regex & markdown table rules for structured AUTOSAR specifications
- LLM-assisted fallback parsing for unstructured narrative text
- Pydantic schema validation
- Entity deduplication
- Optional persistence via storage.database.DatabaseManager
"""

import json
import re
from typing import Any, Optional, Sequence
from core.chunking.models import Chunk
from core.extraction.models import (
    EntityExtractionResult,
    Interface,
    Port,
    SoftwareComponent,
)
from core.llm.base import BaseLLMProvider
from core.logger import get_logger
from storage.database import DatabaseManager

logger = get_logger(__name__)


class EntityExtractor:
    """Lightweight hybrid extractor for AUTOSAR entities."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            llm_provider: Optional LLM provider for unstructured text extraction.
            db_manager: Optional DatabaseManager to persist extracted entities.
        """
        self.llm_provider = llm_provider
        self.db_manager = db_manager

    def extract_from_text(
        self,
        text: str,
        document_id: str,
        page_number: int = 1,
        section: Optional[str] = None,
        use_llm: bool = False,
    ) -> EntityExtractionResult:
        """Extract entities from raw text or markdown.

        Args:
            text: Input text containing AUTOSAR specifications or tables.
            document_id: Identifier of the source document.
            page_number: Source page number.
            section: Source section or heading.
            use_llm: Whether to invoke the LLM provider for narrative extraction.

        Returns:
            Validated EntityExtractionResult container.
        """
        swcs: list[SoftwareComponent] = []
        ports: list[Port] = []
        interfaces: list[Interface] = []

        # 1. Deterministic Table & Rule-Based Extraction
        table_swcs, table_ports, table_ifaces = self._extract_from_tables(
            text=text,
            document_id=document_id,
            page_number=page_number,
            section=section,
        )
        swcs.extend(table_swcs)
        ports.extend(table_ports)
        interfaces.extend(table_ifaces)

        pattern_swcs, pattern_ports, pattern_ifaces = self._extract_from_patterns(
            text=text,
            document_id=document_id,
            page_number=page_number,
            section=section,
        )
        swcs.extend(pattern_swcs)
        ports.extend(pattern_ports)
        interfaces.extend(pattern_ifaces)

        # 2. LLM-assisted extraction if enabled and provider is available
        if use_llm and self.llm_provider is not None:
            llm_swcs, llm_ports, llm_ifaces = self._extract_via_llm(
                text=text,
                document_id=document_id,
                page_number=page_number,
                section=section,
            )
            swcs.extend(llm_swcs)
            ports.extend(llm_ports)
            interfaces.extend(llm_ifaces)

        # 3. Deduplication
        dedup_swcs = self._deduplicate_swcs(swcs)
        dedup_ports = self._deduplicate_ports(ports)
        dedup_ifaces = self._deduplicate_interfaces(interfaces)

        result = EntityExtractionResult(
            document_id=document_id,
            software_components=dedup_swcs,
            ports=dedup_ports,
            interfaces=dedup_ifaces,
        )

        # 4. Optional Storage Persistence
        if self.db_manager is not None:
            self._persist_entities(result)

        return result

    def extract_from_chunks(
        self,
        chunks: Sequence[Chunk],
        document_id: Optional[str] = None,
        use_llm: bool = False,
    ) -> EntityExtractionResult:
        """Extract entities across a list of Chunk objects."""
        if not chunks:
            doc_id = document_id or "unknown_doc"
            return EntityExtractionResult(document_id=doc_id)

        doc_id = document_id or chunks[0].document_id
        all_swcs: list[SoftwareComponent] = []
        all_ports: list[Port] = []
        all_ifaces: list[Interface] = []

        for chunk in chunks:
            res = self.extract_from_text(
                text=chunk.text,
                document_id=doc_id,
                page_number=chunk.page_number,
                section=chunk.section_title,
                use_llm=use_llm,
            )
            all_swcs.extend(res.software_components)
            all_ports.extend(res.ports)
            all_ifaces.extend(res.interfaces)

        result = EntityExtractionResult(
            document_id=doc_id,
            software_components=self._deduplicate_swcs(all_swcs),
            ports=self._deduplicate_ports(all_ports),
            interfaces=self._deduplicate_interfaces(all_ifaces),
        )

        if self.db_manager is not None:
            self._persist_entities(result)

        return result

    def _extract_from_tables(
        self, text: str, document_id: str, page_number: int, section: Optional[str]
    ) -> tuple[list[SoftwareComponent], list[Port], list[Interface]]:
        """Parse standard Markdown tables representing AUTOSAR Ports or SWCs."""
        swcs: list[SoftwareComponent] = []
        ports: list[Port] = []
        interfaces: list[Interface] = []

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        current_swc_context: Optional[str] = None

        # Check if surrounding text specifies an enclosing SWC
        swc_match = re.search(r"\b(SWC_[A-Za-z0-9_]+)\b", text)
        if not swc_match:
            swc_match = re.search(r"(?:Software Component|Component)\s*[:=]?\s*([A-Za-z0-9_]+)", text, re.IGNORECASE)
        if swc_match:
            current_swc_context = swc_match.group(1).strip()

        table_rows: list[list[str]] = []
        for line in lines:
            if line.startswith("|") and line.endswith("|"):
                cells = [c.strip() for c in line[1:-1].split("|")]
                table_rows.append(cells)

        if not table_rows or len(table_rows) < 2:
            return swcs, ports, interfaces

        headers = [h.lower() for h in table_rows[0]]

        # Determine column indices
        port_col = next((i for i, h in enumerate(headers) if "port" in h and "type" not in h), None)
        dir_col = next((i for i, h in enumerate(headers) if any(k in h for k in ("dir", "kind", "type"))), None)
        iface_col = next((i for i, h in enumerate(headers) if "interface" in h), None)
        desc_col = next((i for i, h in enumerate(headers) if "desc" in h), None)
        swc_col = next((i for i, h in enumerate(headers) if "swc" in h or "component" in h), None)

        for row in table_rows[1:]:
            # Skip separator rows like |---|---|
            if any(all(c in "- :" for c in cell) for cell in row):
                continue
            if port_col is not None and port_col < len(row):
                port_name = row[port_col].strip()
                if port_name and re.match(r"^[A-Za-z0-9_]+$", port_name):
                    direction = row[dir_col].strip() if dir_col is not None and dir_col < len(row) else "Provided"
                    iface_name = row[iface_col].strip() if iface_col is not None and iface_col < len(row) else None
                    desc = row[desc_col].strip() if desc_col is not None and desc_col < len(row) else None
                    row_swc = row[swc_col].strip() if swc_col is not None and swc_col < len(row) else current_swc_context

                    try:
                        p = Port(
                            name=port_name,
                            direction=direction,
                            swc_parent=row_swc,
                            interface_name=iface_name if iface_name and iface_name != "-" else None,
                            description=desc if desc and desc != "-" else None,
                            document_id=document_id,
                            page_number=page_number,
                            section=section,
                        )
                        ports.append(p)
                    except Exception as err:
                        logger.debug("Failed to validate Port row %s: %s", row, err)

                    if iface_name and iface_name != "-" and re.match(r"^[A-Za-z0-9_]+$", iface_name):
                        try:
                            iface_type = "SenderReceiverInterface"
                            if "cs" in iface_name.lower() or "client" in iface_name.lower():
                                iface_type = "ClientServerInterface"
                            iface = Interface(
                                name=iface_name,
                                interface_type=iface_type,
                                description=f"Interface linked to port {port_name}",
                                document_id=document_id,
                                page_number=page_number,
                                section=section,
                            )
                            interfaces.append(iface)
                        except Exception as err:
                            logger.debug("Failed to validate Interface from row %s: %s", row, err)

        return swcs, ports, interfaces

    def _extract_from_patterns(
        self, text: str, document_id: str, page_number: int, section: Optional[str]
    ) -> tuple[list[SoftwareComponent], list[Port], list[Interface]]:
        """Deterministic regex extraction for standard AUTOSAR entity naming and patterns."""
        swcs: list[SoftwareComponent] = []
        ports: list[Port] = []
        interfaces: list[Interface] = []

        # 1. SWCs: patterns like `SWC_BatteryManager`, `Software Component: BMS_Core`
        swc_patterns = [
            r"\b(SWC_[A-Za-z0-9_]+)\b",
            r"(?:Software Component|SWC)\s*[:=]\s*([A-Za-z0-9_]+)",
        ]
        detected_swc_names: set[str] = set()
        for pat in swc_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                name = m.group(1).strip()
                if name and name not in detected_swc_names:
                    detected_swc_names.add(name)
                    swc_type = "ApplicationSwComponentType"
                    if "sensor" in name.lower() or "actuator" in name.lower():
                        swc_type = "SensorActuatorSwComponentType"
                    elif "service" in name.lower():
                        swc_type = "ServiceSwComponentType"

                    try:
                        swcs.append(
                            SoftwareComponent(
                                name=name,
                                component_type=swc_type,
                                description=f"Detected software component {name}",
                                document_id=document_id,
                                page_number=page_number,
                                section=section,
                            )
                        )
                    except Exception as err:
                        logger.debug("Error creating SoftwareComponent %s: %s", name, err)

        # Primary enclosing SWC if any was found
        primary_swc = next(iter(detected_swc_names), None)

        # 2. Ports: P_*, R_*, PR_* or explicit Port definitions
        port_pattern = r"\b((?:P|R|PR)_[A-Za-z0-9_]+)\b"
        for m in re.finditer(port_pattern, text):
            pname = m.group(1).strip()
            direction = "Provided" if pname.startswith("P_") else ("Required" if pname.startswith("R_") else "ProvidedRequired")
            try:
                ports.append(
                    Port(
                        name=pname,
                        direction=direction,
                        swc_parent=primary_swc,
                        interface_name=None,
                        description=f"Port with direction {direction}",
                        document_id=document_id,
                        page_number=page_number,
                        section=section,
                    )
                )
            except Exception as err:
                logger.debug("Error creating Port %s: %s", pname, err)

        # 3. Interfaces: If_*, Interface: If_*
        iface_patterns = [
            r"\b(If_[A-Za-z0-9_]+)\b",
            r"\b(I[A-Z][a-zA-Z0-9_]+Interface)\b",
            r"(?:Interface)\s*[:=]\s*([A-Za-z0-9_]+)",
        ]
        detected_ifaces: set[str] = set()
        for pat in iface_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                iname = m.group(1).strip()
                if iname and iname not in detected_ifaces:
                    detected_ifaces.add(iname)
                    iface_type = "ClientServerInterface" if "cs" in iname.lower() or "clientserver" in iname.lower() else "SenderReceiverInterface"
                    try:
                        interfaces.append(
                            Interface(
                                name=iname,
                                interface_type=iface_type,
                                description=f"Interface {iname}",
                                document_id=document_id,
                                page_number=page_number,
                                section=section,
                            )
                        )
                    except Exception as err:
                        logger.debug("Error creating Interface %s: %s", iname, err)

        return swcs, ports, interfaces

    def _extract_via_llm(
        self, text: str, document_id: str, page_number: int, section: Optional[str]
    ) -> tuple[list[SoftwareComponent], list[Port], list[Interface]]:
        """LLM-assisted structured extraction for complex narrative passages."""
        swcs: list[SoftwareComponent] = []
        ports: list[Port] = []
        interfaces: list[Interface] = []

        if not self.llm_provider:
            return swcs, ports, interfaces

        system_prompt = (
            "You are an expert AUTOSAR High-Level Design (HLD) architecture parser.\n"
            "Extract Software Components, Ports, and Interfaces from the provided text.\n"
            "Output strictly valid JSON with this exact schema:\n"
            "{\n"
            '  "software_components": [{"name": "string", "component_type": "string", "description": "string"}],\n'
            '  "ports": [{"name": "string", "direction": "Provided|Required|ProvidedRequired", "swc_parent": "string", "interface_name": "string", "description": "string"}],\n'
            '  "interfaces": [{"name": "string", "interface_type": "SenderReceiverInterface|ClientServerInterface|ParameterInterface", "description": "string"}]\n'
            "}\n"
            "Do not include markdown code block formatting or explanations outside the JSON."
        )

        user_prompt = f"Text to analyze:\n---\n{text}\n---"

        try:
            response = self.llm_provider.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            raw_content = response.content.strip()
            # Clean markdown codeblocks if LLM returned them
            cleaned = re.sub(r"^```(?:json)?", "", raw_content, flags=re.MULTILINE)
            cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
            data = json.loads(cleaned)

            for item in data.get("software_components", []):
                try:
                    swcs.append(
                        SoftwareComponent(
                            name=item.get("name", "").strip(),
                            component_type=item.get("component_type") or "ApplicationSwComponentType",
                            description=item.get("description"),
                            document_id=document_id,
                            page_number=page_number,
                            section=section,
                        )
                    )
                except Exception as err:
                    logger.debug("Invalid SWC from LLM: %s", err)

            for item in data.get("ports", []):
                try:
                    ports.append(
                        Port(
                            name=item.get("name", "").strip(),
                            direction=item.get("direction", "Provided"),
                            swc_parent=item.get("swc_parent"),
                            interface_name=item.get("interface_name"),
                            description=item.get("description"),
                            document_id=document_id,
                            page_number=page_number,
                            section=section,
                        )
                    )
                except Exception as err:
                    logger.debug("Invalid Port from LLM: %s", err)

            for item in data.get("interfaces", []):
                try:
                    interfaces.append(
                        Interface(
                            name=item.get("name", "").strip(),
                            interface_type=item.get("interface_type") or "SenderReceiverInterface",
                            description=item.get("description"),
                            document_id=document_id,
                            page_number=page_number,
                            section=section,
                        )
                    )
                except Exception as err:
                    logger.debug("Invalid Interface from LLM: %s", err)

        except Exception as err:
            logger.warning("LLM-assisted entity extraction failed: %s", err)

        return swcs, ports, interfaces

    def _deduplicate_swcs(self, swcs: list[SoftwareComponent]) -> list[SoftwareComponent]:
        seen: dict[str, SoftwareComponent] = {}
        for s in swcs:
            key = (s.name.lower(), s.document_id)
            if key not in seen:
                seen[key] = s
            else:
                existing = seen[key]
                if not existing.description and s.description:
                    existing.description = s.description
        return list(seen.values())

    def _deduplicate_ports(self, ports: list[Port]) -> list[Port]:
        # Key by (port_name.lower(), document_id) to avoid duplicate entries when
        # table extraction and pattern extraction find the same port with or without full context
        seen: dict[tuple[str, str], Port] = {}
        for p in ports:
            key = (p.name.lower(), p.document_id)
            if key not in seen:
                seen[key] = p
            else:
                existing = seen[key]
                if not existing.swc_parent and p.swc_parent:
                    existing.swc_parent = p.swc_parent
                if not existing.interface_name and p.interface_name:
                    existing.interface_name = p.interface_name
                if (not existing.description or "Port with direction" in existing.description) and p.description:
                    existing.description = p.description
        return list(seen.values())

    def _deduplicate_interfaces(self, ifaces: list[Interface]) -> list[Interface]:
        seen: dict[tuple[str, str], Interface] = {}
        for i in ifaces:
            key = (i.name.lower(), i.document_id)
            if key not in seen:
                seen[key] = i
            else:
                existing = seen[key]
                if not existing.description and i.description:
                    existing.description = i.description
        return list(seen.values())

    def _persist_entities(self, result: EntityExtractionResult) -> None:
        """Save extracted entities to database."""
        if not self.db_manager:
            return
        try:
            self.db_manager.save_software_components(result.software_components)
            self.db_manager.save_ports(result.ports)
            self.db_manager.save_interfaces(result.interfaces)
            logger.info(
                "Persisted %d entities for doc '%s' into SQLite database",
                result.total_entities,
                result.document_id,
            )
        except Exception as err:
            logger.error("Failed to persist entities to database: %s", err)
