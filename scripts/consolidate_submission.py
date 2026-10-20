"""Consolidate the formal capstone submission PDFs and remove duplicate files.

Merges the already-generated Times New Roman 12 pt PDFs; does not modify
working application source files, runtime databases, or student information.
"""
from pathlib import Path
import shutil
from pypdf import PdfReader, PdfWriter

ROOT=Path(__file__).resolve().parents[1]
MERGE={
    "Synopsis/Project_Synopsis.pdf": [
        "Synopsis/SYNOPSIS.pdf",
    ],
    "Documentation/Technical_Report.pdf": [
        "Documentation/TECHNICAL_REPORT_SOURCE.pdf",
        "Documentation/ARCHITECTURE_AND_DATA_FLOW.pdf",
        "Documentation/IMPLEMENTATION_MODULE_REFERENCE.pdf",
        "Documentation/PREPROCESSING_AND_INPUT_DATA.pdf",
        "Documentation/AI_ML_ALGORITHMS_AND_METRICS.pdf",
        "Documentation/RESPONSIBLE_AI_AND_SECURITY.pdf",
        "Documentation/SETUP_AND_DEMO_RUNBOOK.pdf",
        "Documentation/VIVA_PREPARATION.pdf",
        "Documentation/REPOSITORY_AUDIT_CHECKLIST.pdf",
        "Declarations/EXTERNAL_SOURCES_REGISTER.pdf",
    ],
    "Model_Prompts_Config/Model_and_Prompt_Configuration.pdf": [
        "Model_Prompts_Config/MODELS_AND_AI_METHODS.pdf",
        "Model_Prompts_Config/CONFIGURATION_AND_RETRIEVAL_PARAMETERS.pdf",
        "Model_Prompts_Config/PROMPT_DESIGN_AND_EXAMPLES.pdf",
    ],
    "Evaluation_Results/Evaluation_Report.pdf": [
        "Evaluation_Results/EVALUATION_PROTOCOL.pdf",
        "Evaluation_Results/EVALUATION_STATUS.pdf",
        "Evaluation_Results/GOLDEN_QUESTIONS.pdf",
    ],
    "Declarations/Academic_Integrity_Declaration.pdf": [
        "Declarations/ACADEMIC_INTEGRITY_DECLARATION_TEMPLATE.pdf",
    ],
    "Declarations/AI_Tool_Usage_Declaration.pdf": [
        "Declarations/AI_TOOL_USAGE_DECLARATION_TEMPLATE.pdf",
    ],
}
KEEP={
    "Input_Data/README.md",
    "Code/README.md",
    "Video/README.md",
    "Model_Prompts_Config/SAFE_CONFIG_REFERENCE.env.example",
    "Evaluation_Results/RESULTS_TO_FILL.csv",
}
CLEAN_DIRS=["Synopsis","Documentation","Model_Prompts_Config","Evaluation_Results","Declarations"]

def main():
    for target, inputs in MERGE.items():
        for item in inputs:
            if not (ROOT/item).is_file():
                raise FileNotFoundError("Cannot consolidate: missing source "+item)
        output=ROOT/target
        output.parent.mkdir(parents=True,exist_ok=True)
        writer=PdfWriter()
        for item in inputs:
            reader=PdfReader(str(ROOT/item))
            if not reader.pages:
                raise ValueError("PDF has no pages: "+item)
            writer.append(reader)
        with output.open("wb") as handle:
            writer.write(handle)
        print(target, "merged from",len(inputs),"documents;",len(writer.pages),"pages")
    keep=set(MERGE)|KEEP
    for folder in CLEAN_DIRS:
        for file in (ROOT/folder).iterdir():
            if file.is_file() and file.relative_to(ROOT).as_posix() not in keep:
                file.unlink()
                print("Removed duplicate:",file.relative_to(ROOT))
    report=ROOT/"SUBMISSION_README.md"
    report.write_text(
        "# AUTOSAR HLD AI Assistant — Case Study 1 Submission\n\n"
        "Required academic folder structure (exactly eight folders):\n\n"
        "| Folder | Contents |\n|---|---|\n"
        "| Synopsis/ | Project_Synopsis.pdf |\n"
        "| Input_Data/ | README.md (sample PDF under repository data/samples/) |\n"
        "| Code/ | README.md (working Python source remains at repository root) |\n"
        "| Model_Prompts_Config/ | Model_and_Prompt_Configuration.pdf and SAFE_CONFIG_REFERENCE.env.example |\n"
        "| Evaluation_Results/ | Evaluation_Report.pdf and RESULTS_TO_FILL.csv |\n"
        "| Documentation/ | Technical_Report.pdf (comprehensive consolidated report and appendices) |\n"
        "| Video/ | README.md (demo deferred) |\n"
        "| Declarations/ | Academic_Integrity_Declaration.pdf and AI_Tool_Usage_Declaration.pdf |\n\n"
        "Formal PDF documents use Times New Roman at 12 pt. Declaration templates "
        "remain unsigned and require student review. Evaluation records distinguish "
        "historical observations from verified real-model runs.\n\n"
        "The live application source code remains unchanged outside these submission "
        "folders. Do not upload API credentials, .env, private OEM data or local indexes.\n",
        encoding="utf-8"
    )
    for folder in CLEAN_DIRS:
        files=[p.name for p in (ROOT/folder).iterdir() if p.is_file()]
        print(folder, "=>", sorted(files))
    assert len(list((ROOT/"Synopsis").iterdir()))==1
    assert len(list((ROOT/"Documentation").iterdir()))==1
    assert len(list((ROOT/"Declarations").iterdir()))==2
    assert len(list((ROOT/"Model_Prompts_Config").iterdir()))==2

if __name__=="__main__":
    main()
