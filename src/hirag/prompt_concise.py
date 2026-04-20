from copy import deepcopy

from .prompt_baseline import PROMPTS as BASELINE_PROMPTS


LEAN_HI_ENTITY_EXTRACTION = """
Extract entities from the text.

Rules:
- Only extract entities explicitly supported by the text.
- Use one of these types: [{entity_types}]. Use normal_entity only when no listed type fits.
- Keep each entity_description to one short factual sentence.
- Output only records in this format:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
- Separate records with {record_delimiter}.
- End with {completion_delimiter}.

Example 1:
Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. Then Taylor paused beside Jordan and observed the device with reverence.
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person reacting to tension in the group."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who studies the device closely."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person interacting with Taylor around the device."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is a technology under discussion."){completion_delimiter}

Example 2:
Entity_types: [person, technology, mission, organization, location]
Text:
Communications with Washington buzzed in the background as the team moved to address the warning. Operation: Dulce had evolved into an active mission.
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is the source of background communications."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is the group handling the warning."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is the mission guiding the team's actions."){completion_delimiter}

Real Data:
Entity_types: {entity_types}
Text: {input_text}
Output:
"""

ULTRA_LEAN_HI_ENTITY_EXTRACTION = """
Extract all explicit entities from the text.

Rules:
- Use one of these types: [{entity_types}]. Use normal_entity only if needed.
- Keep entity_description very short.
- Output only:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
- Separate records with {record_delimiter}.
- End with {completion_delimiter}.

Example:
Entity_types: [person, technology, mission, organization, location]
Text:
Taylor paused beside Jordan and observed the device while Alex watched.
Output:
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor observes the device."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan stands near the device."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex watches the interaction."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is the object being observed."){completion_delimiter}

Real Data:
Entity_types: {entity_types}
Text: {input_text}
Output:
"""

LEAN_HI_RELATION_EXTRACTION = """
Extract explicit relationships among the provided entities.

Rules:
- Only include clearly supported relationships.
- Keep relationship_description to one short factual sentence.
- Output only records in this format:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
- Use an integer or short decimal strength.
- Separate records with {record_delimiter}.
- End with {completion_delimiter}.

Example 1:
Entities: ["Alex", "Taylor", "Jordan", "The Device"]
Text:
Alex watched as Taylor paused beside Jordan and observed the device with reverence.
Output:
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex watches Taylor during the interaction."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor stands beside Jordan while examining the device."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor examines the device directly."{tuple_delimiter}9){completion_delimiter}

Example 2:
Entities: ["Washington", "Operation: Dulce", "The team"]
Text:
Communications with Washington buzzed in the background as the team advanced Operation: Dulce.
Output:
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is carrying out Operation: Dulce."{tuple_delimiter}9){completion_delimiter}

Real Data:
Entities: {entities}
Text: {input_text}
Output:
"""

ULTRA_LEAN_HI_RELATION_EXTRACTION = """
Extract explicit relationships among the provided entities.

Rules:
- Only include clearly supported relationships.
- Keep relationship_description very short.
- Output only:
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
- Separate records with {record_delimiter}.
- End with {completion_delimiter}.

Example:
Entities: ["Taylor", "Jordan", "The Device"]
Text:
Taylor paused beside Jordan and observed the device.
Output:
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor stands beside Jordan."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor observes the device."{tuple_delimiter}9){completion_delimiter}

Real Data:
Entities: {entities}
Text: {input_text}
Output:
"""

LEAN_SUMMARIZE_ENTITY_DESCRIPTIONS = """You will merge duplicate descriptions for the same entity or relation.

Rules:
- Produce one concise third-person summary.
- Keep the summary to at most 2 short sentences.
- Preserve important concrete facts when they are consistent.

Data:
Entities: {entity_name}
Description List: {description_list}
Output:
"""

ULTRA_LEAN_SUMMARIZE_ENTITY_DESCRIPTIONS = """Merge the descriptions into one short third-person summary.

Rules:
- Keep it factual and brief.
- Use at most 1 short sentence.

Data:
Entities: {entity_name}
Description List: {description_list}
Output:
"""

LEAN_SUMMARY_CLUSTERS = """Summarize a cluster of entities into one higher-level attribute entity.

Rules:
- Generate exactly 1 attribute entity.
- The entity_type must be one of [{meta_attribute_list}] or normal_entity.
- Keep the entity_description to one short factual sentence.
- For each source entity, add a short relationship to the summary entity.
- Do not create relationships between summary entities.
- Output only:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
- Separate records with {record_delimiter}.
- End with {completion_delimiter}.

Example:
Input:
Meta attribute list: ["company", "location"]
Entity description list: [("Instagram", "Instagram is a software developed by Meta."), ("Facebook", "Facebook is owned by Meta."), ("WhatsApp", "WhatsApp is a messaging app of Meta.")]
Output:
("entity"{tuple_delimiter}"Meta"{tuple_delimiter}"company"{tuple_delimiter}"Meta is the parent company connecting these products."){record_delimiter}
("relationship"{tuple_delimiter}"Instagram"{tuple_delimiter}"Meta"{tuple_delimiter}"Instagram belongs to Meta."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Facebook"{tuple_delimiter}"Meta"{tuple_delimiter}"Facebook belongs to Meta."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"WhatsApp"{tuple_delimiter}"Meta"{tuple_delimiter}"WhatsApp belongs to Meta."{tuple_delimiter}8){completion_delimiter}

Real Data:
Input:
Meta attribute list: {meta_attribute_list}
Entity description list: {entity_description_list}
Output:
"""

ULTRA_LEAN_SUMMARY_CLUSTERS = """Summarize the cluster into one higher-level attribute entity.

Rules:
- Generate exactly 1 attribute entity.
- Keep all descriptions short.
- Output only entity and relationship records in the required format.
- Separate records with {record_delimiter}.
- End with {completion_delimiter}.

Real Data:
Input:
Meta attribute list: {meta_attribute_list}
Entity description list: {entity_description_list}
Output:
"""

LEAN_COMMUNITY_REPORT = """You are given entities and relationships for one community.

Return a JSON object with this shape:
{
  "title": "...",
  "summary": "...",
  "findings": [
    {"summary": "...", "explanation": "..."}
  ]
}

Rules:
- Ground everything in the provided data.
- Keep the summary to 2-3 sentences.
- Return at most 3 findings.
- Keep each explanation concise.
- Do not add unsupported facts.

Text:
{input_text}
Output:
"""

ULTRA_LEAN_COMMUNITY_REPORT = """Return a grounded JSON object for the community.

Schema:
{
  "title": "...",
  "summary": "...",
  "findings": [
    {"summary": "...", "explanation": "..."}
  ]
}

Rules:
- Keep the summary short.
- Return at most 2 findings.
- Keep each explanation brief.
- Use only supported facts.

Text:
{input_text}
Output:
"""


def build_lean_prompts() -> dict[str, object]:
    prompts = deepcopy(BASELINE_PROMPTS)
    prompts["hi_entity_extraction"] = LEAN_HI_ENTITY_EXTRACTION
    prompts["hi_relation_extraction"] = LEAN_HI_RELATION_EXTRACTION
    prompts["summary_clusters"] = LEAN_SUMMARY_CLUSTERS
    prompts["community_report"] = LEAN_COMMUNITY_REPORT
    prompts["summarize_entity_descriptions"] = LEAN_SUMMARIZE_ENTITY_DESCRIPTIONS
    return prompts


def build_ultra_lean_prompts() -> dict[str, object]:
    prompts = deepcopy(BASELINE_PROMPTS)
    prompts["hi_entity_extraction"] = ULTRA_LEAN_HI_ENTITY_EXTRACTION
    prompts["hi_relation_extraction"] = ULTRA_LEAN_HI_RELATION_EXTRACTION
    prompts["summary_clusters"] = ULTRA_LEAN_SUMMARY_CLUSTERS
    prompts["community_report"] = ULTRA_LEAN_COMMUNITY_REPORT
    prompts["summarize_entity_descriptions"] = ULTRA_LEAN_SUMMARIZE_ENTITY_DESCRIPTIONS
    return prompts

