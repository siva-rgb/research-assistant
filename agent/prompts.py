"""
All prompt templates for the Research Agent.
Design Rule:
- Every Prompt is a plain siring with {variable} placeholder
- No prompt logic lives in the node call format_prompt() and pass kwargs
"""

SYSTEM_PROMPT= """You are a smart research assistant. Your job is to gather \
    information from multiple source, synthesize it accuratly and procduice well cited research report.
    Rule you always follow:
    -Be factual. Never invent source, statistics, or claims.
    -Be concise. Produce exactly the output format requested- no extra commentry.
    -Be honest about uncerteainty. If sources confilicts or information is incomplete,\
        say so explicitly.
    - Never reforuse a search based on the grounds of difficulty, work with what you find."""

INTENT_EXTRACTION_PROMPT= """
Analyze the following research query and extract structured intent
Query: {query}

Respons with a JSON object and nothing else - no markdoen, no explanation:
{{
"intent": "<one clear paragraph restating the research goal in your own words>",
"research_scope": "<one of: Narrow | Moderate | Broad>",
"key_concepts": ["<concept1>", "<concept2>", "..."]
"reasoing": "<one sentence explaining your scope classification>"
}}

Scope defination:
-Narrow: a specific factual question with a definative answer (1-2 search needed)
-Moderate: a topic requiring multiple resource and some synthesis (3-5 searches)
-broad: a complex topic requiring deep research and extensive synthesis (5+ searches)

"""

PLLANER_PROMPT= """
Create a step-by-step researches plan for the following goal.
Research Goal:{intent}
Scope:{research_scope}
Key concepts: {key_concepts}

Available Tools:
{tool_descriptions}

Respond with JSON object and nothing else- no markdown, no explanation:
{{
"subtasks": [
    {{
    "step_index":0,
    "task_type": "search",
    "description": "<what this step is trying to find out>",
    "query": "<The Exact search query or URL to use>",
    "status": "pending"
    }}
],
"total_steps": <integer>
}}

Planning Rules:
-Narrow Scope- 2-3 subtasks maximum
-Moderate Scope- 3-5 subtasks
-Broad Scope: 5-7 Subtasks maximum(never more than 7)
-First subtask should always be a braod overview search
-Last subtask before synthesis should verify or cross-check a keyt claim
-Use read_document only if a specific url is already known and critical
"""

QUERY_REFINEMENT_PROMPT= """Given the current research findings and the next subtask to execute,\
    Generat the best possible search query.
    
    Original research goal: {intent}
    Next subtask: {subtask_description}
    Suggest query for plan: {planned_query}
    Findings so far (summary): {finding_summary}
    
    If the suggested query is already good, return it unchanges.
    If findings suggest a more specific or better-targeted query, improve it.
    Respond with a JSON object and nothing else:
    {{
    "query": "<the final search query to use>",
    "reasoning": "<one sentence explaining any changes made>"
    }}"""

REFLECTION_PROMPT="""You are evaluating the quality and completness of research findings.

Original research goal:{intent}
Research plan had {total_steps} steps. Steps completed:{steps_completed} 

Findings collected so far:
{findings_summary}

Evaluate the findings agains the research goal and respond with a JSON object nothing else
{{
"critique": "<what is missing, weak, or contradictory in the current findings>",
"decision": "<one of: accept | revise | abort>"
"confidence_score": <float 0.0-0.1 representing completness>
"additional_queries": ["<query 1 if decision is revise>"],
"reasoning": "<One sentence explaining the decision>"
}}

Decision Rules:
-accept: findings sufficiently address research goal (confidence >=0.7)
-revise: importnat gaps exist that additional researches fill (confidence 0.4-0.9)
-abort: ther research goal cannot be addressed with available information\
(confidence <0.4 after multiple attempts, or fundamentally unanswerable)
"""

SYNTHESIS_PROMPT= """
Write a research report based on the following findings.

Research goal:{intent}
Source and findings: {formatted_findings}

Write a well structured research report that:
1. Open with a direct answer to the research question (2-3 sentences)
2. Presents supporting evidence organized by theme (not by source)
3. Acknowledge any contradiction or uncertainties found
4. Close with a brief summary of key takeaways

Cite sources inline using [Source: url] format.
Be factual. Do not add inforamtion beyond what findings contain.
Taget lengtrh:300-600 words
"""

def format_findings_for_prompt(
        search_results: list[dict],
        document_read: list[dict],
        max_chars: int=6000
)-> str:
    """Convets state. search_results and state.documents_read into a foramted string suitable for injection of into synthesis and reflection prompts.
    Startefy: Include all search reasult first (they're short), then append document content until we hit the cap.
    """

    sections=[]
    total_chars= 0

    if search_results:
        sections.append("=== SEARCH RESULTS ===")
        for i, result in enumerate(search_results,1):
            entry=(f"[{i}] {result['title']}\n"
                   f"URL: {result['url']}\n"
                   f"Relevance: {result['source']}\n"
                   f"Content: {result['content']}\n")
            total_chars +=len(entry)
            sections.append(entry)

    if document_read:
        sections.append("=== FULL DOCUMENT READ")
        for doc in document_read:
            remaining= max_chars- total_chars
            if remaining<=0:
                sections.append("[Additional document omited - contxt limit reached]")
                break
            content= doc.get("content","")[:remaining]
            entry= (
                f"URL: {doc['url']}\n"
                f"Content:\n{content}\n"

            )
            total_chars+=len(entry)
            sections.append(entry)
    return "\n".join(sections) if sections else "No findings collected yet"

def format_findings_summary(search_result:list[dict],
                            max_chars: int=2000):
    """
    Shorter versin of findings for use in reflection and query refinement prompts.
    only search reasult title and snippets - no full documemts.
    """
    if not search_result:
        return "no findings"
    
    lines=[]
    total=0
    for r in search_result:
        line= f"- {r['title']}: {r['content'][:200]}"
        total+= len(line)
        if total > max_chars:
            lines.append("...[truncated]")
            break
        lines.append(line)

    return '\n'.join(lines)

