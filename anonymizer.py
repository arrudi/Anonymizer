#!/usr/bin/env python3

import os
import argparse
import yaml
from tqdm import tqdm
from pathlib import Path
from typing import Any
import fitz  # PyMuPDF
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

CONFIG = "config.yaml"
YAM_FILE = "recognizers.yml"

def init_analyzer(model: str)-> AnalyzerEngine:
    #Available models at https://spacy.io/models
    language =  model[:2]
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": f"{language}", "model_name": f"{model}"}, # provided Model
            #{"lang_code": "en", "model_name": "en_core_web_lg"}, # English Large Model
            #{"lang_code": "en", "model_name": "en_core_web_trf"}, # English Transformer
            #{"lang_code": "es", "model_name": "es_core_news_lg"}  # Spanish Large Model
            # You can easily add more here, e.g., {"lang_code": "fr", "model_name": "fr_core_news_lg"}
        ],
    }

    # Initialize the provider with our explicit spaCy mapping
    provider = NlpEngineProvider(nlp_configuration=configuration)
    concrete_nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=concrete_nlp_engine)
    
    load_recognizers(analizer=analyzer, file=CONFIG)
    return analyzer


def load_exclusions(file: str | Path)-> list[str]:
    yaml_path = Path(file)
    with yaml_path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    exclusion_list = config.get("exclusions", [])
    if not isinstance(exclusion_list, list):
        raise ValueError("'exclusions' must be a list")
    
    return exclusion_list
    

def load_recognizers(analizer: AnalyzerEngine, file: str | Path):

    yaml_path = Path(file)
    with yaml_path.open("r", encoding="utf-8") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    recognizer_configs = config.get("recognizers", [])
    if not isinstance(recognizer_configs, list):
        raise ValueError("'recognizers' must be a list")
    
    for item in recognizer_configs:
        recognizer_name = require_string(item, "recognizer_name")
        supported_entity = require_string(item, "supported_entity")
        supported_language = item.get("supported_language", "en")
        context = item.get("context", [])
        pattern_configs = item.get("patterns", [])
        if not isinstance(pattern_configs, list) or not pattern_configs:
            raise ValueError(f"Recognizer '{recognizer_name}' must define at least one pattern")
        
        if not isinstance(context, list):
            raise ValueError(f"Recognizer '{recognizer_name}' has invalid 'context'; expected list")

        patterns = []
            
        for pattern_config in pattern_configs:
            pattern_name = require_string(pattern_config, "name")
            regex = require_string(pattern_config, "regex")
            score = require_score(pattern_config, "score")
            patterns.append(Pattern(name=pattern_name,regex=regex,score=score,))

        pattern = PatternRecognizer(name=recognizer_name,
                                        supported_entity=supported_entity,
                                        supported_language=supported_language,
                                        patterns=patterns,
                                        context=context,)
        analizer.registry.add_recognizer(recognizer=pattern)

    print(f"Added {len(recognizer_configs)} recognizer patterns from external file") 


def load_entities(filename: str | Path) -> list[str]:
    yaml_path = Path(filename)
    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    entities = { recognizer["supported_entity"] for recognizer in config.get("recognizers", []) if "supported_entity" in recognizer }
    return sorted(entities)


def require_string(config: dict[str, Any], field_name: str) -> str:

    value = config.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid string field: {field_name}")
    return value

def require_score(config: dict[str, Any], field_name: str) -> float:
    value = config.get(field_name)
    if not isinstance(value, int | float):
        raise ValueError(f"Missing or invalid numeric field: {field_name}")
    value = float(value)

    if not 0 <= value <= 1:
        raise ValueError(f"Score must be between 0 and 1: {value}")
    return value


def anonymize_file(source_path: Path, destination_path: Path, analyzer: AnalyzerEngine, language: str) -> None:

    #print(f"Translating {language} prefix {prefix}: {source_path} -> {destination_path}")
    # Standard entities supported by Presidio across English/Spanish models
    target_entities = [ "PERSON", "LOCATION", "ORGANIZATION",  "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME",] 
    target_entities = target_entities + load_entities(CONFIG)
    
    exclusion_allow_list = load_exclusions(CONFIG)
    
    doc = fitz.open(source_path)

    pii_found = False

    for page in tqdm(doc, desc=f" anonymizing {source_path.name}", unit="page", colour="green",):
        
        text = page.get_text()
        if not text.strip():
            continue

        results = analyzer.analyze(text=text, language=language, entities=target_entities, allow_list=exclusion_allow_list)
        if not results:
            continue

        pii_found = True

        for result in results:
            sensitive_string = text[result.start:result.end].strip()
            if not sensitive_string or len(sensitive_string) < 2:
                continue

            # Geolocate the coordinates of the sensitive text fragment on the page
            text_instances = page.search_for(sensitive_string)
            
            for inst in text_instances:
                # Add redaction: changes inner text bytes to 'X's and preps black fill box
                page.add_redact_annot(
                    inst, 
                    text="X" * len(sensitive_string), 
                    fill=(0, 0, 0)
                )
        
        # Apply changes to the page layer
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    if pii_found:
        # Save with optimization flags to wipe absolute trace of historical data streams
        doc.save(destination_path, garbage=4, deflate=True)
        tqdm.write(f"File {source_path.name} securely annonymized to {destination_path.name}")
    else:
        tqdm.write(f"No PII identified in {source_path.name}. Copying original file format.")
        #print(f"⚠️ No PII flagged in {source_path.name}. Copying original file format.")
        doc.save(destination_path)
        
    doc.close()



def anonymize_directory(source_dir: Path, prefix: str, output_dir: Path | None, analyzer: AnalyzerEngine, language: str) -> None:

    #Process every file inside a directory recursively.
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        #destination_root = output_dir / f"{prefix}{source_dir.name}"
        destination_root = output_dir

    else:
        destination_root = source_dir.with_name(f"{prefix}{source_dir.name}")

    destination_root.mkdir(parents=True, exist_ok=True)
    
    files = source_dir.rglob("*.pdf")
    list_files = list(files)
    print(f"Found {len(list_files)} files")

    for source_file in list_files:
        if not source_file.is_file():
            continue

        relative_path = source_file.relative_to(source_dir)
        destination_file = destination_root / f"{prefix}{relative_path}"
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        anonymize_file(source_file, destination_file, analyzer, language)




def main() -> None:

    parser = argparse.ArgumentParser(description="Contract Anonymizer Tool.")

    parser.add_argument( "source", type=Path, help="Contract File or directory to process.",)
    parser.add_argument( "-p", "--prefix", default="ANON_", help="Prefix for created files and folders. Default: ANON_",)
    parser.add_argument( "-o", "--output-dir", type=Path, help="Directory where output is stored. ",)
    parser.add_argument( "-m", "--model", default="en_core_web_trf", help="Recognition model at https://spacy.io/models. Install python3 -m spacy download en_core_web_lg")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None

    if not source.exists():
        parser.error(f"Source does not exist: {source}")

    analyzer = init_analyzer(args.model)
    language = args.model[:2]

    if source.is_dir():
        anonymize_directory(source, args.prefix, output_dir, analyzer, language)

    elif source.is_file():
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            destination = output_dir / f"{args.prefix}{source.name}"
        else:
            destination = source.with_name(f"{args.prefix}{source.name}")
        
        anonymize_file(source, destination, analyzer, language)

    else:
        parser.error(f"Source is neither a regular file nor a directory: {source}")

    print("Anonymizer completed.")



if __name__ == "__main__":
    main()