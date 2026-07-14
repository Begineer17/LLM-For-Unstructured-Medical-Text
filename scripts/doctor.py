"""Environment and pipeline smoke check."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clinical_nlp_pipeline as c

sample = "The note contains no configured terminology."
icd, rx = c.Ontology(None, kind='icd'), c.Ontology(None, kind='rxnorm')
data = c.process_document(sample, icd, rx, require_llm=False); errors = c.validate_pair(sample, data)
print(json.dumps({'python':sys.version.split()[0], 'rapidfuzz':c.fuzz is not None, 'smoke_entities':len(data), 'errors':errors}, ensure_ascii=False, indent=2))
sys.exit(1 if errors else 0)
