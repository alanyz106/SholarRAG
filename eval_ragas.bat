@echo off
REM NexusRAG RAGAS Evaluation Script
REM ==================================
REM
REM Available commands:
REM   generate   - Generate synthetic testset
REM   chat       - Run testset through chat (saves intermediate results)
REM   rule-based - Evaluate with rule-based metrics (fast)
REM   ragas-llm  - Evaluate with RAGAS LLM metrics (requires judge LLM)
REM   evaluate   - Legacy: run chat + both metrics
REM   all        - All-in-one: generate + chat + both metrics
REM
REM Output files:
REM   ragas_testset.json           - Generated test questions
REM   ragas_chat_results.json      - Intermediate chat results
REM   ragas_rule_based_results.json - Rule-based metrics evaluation
REM   ragas_llm_results.json       - RAGAS LLM metrics evaluation
REM   ragas_eval_results.json      - Combined results (for 'all' command)

REM ===== Recommended workflow (separate evaluations) =====

REM Step 1: Generate testset
python backend/scripts/eval_ragas_synthetic.py generate --workspace 11 --size 15 --doc-ids 26

REM Step 2: Run chat to get answers (saves intermediate results)
python backend/scripts/eval_ragas_synthetic.py chat --workspace 11

REM Step 3a: Evaluate with rule-based metrics (fast, no LLM calls)
python backend/scripts/eval_ragas_synthetic.py rule-based --workspace 11

REM Step 3b: Evaluate with RAGAS LLM metrics (separate, can retry independently)
python backend/scripts/eval_ragas_synthetic.py ragas-llm --workspace 11

REM ===== Alternative: All-in-one =====
REM python backend/scripts/eval_ragas_synthetic.py all --workspace 11 --size 10 --doc-ids 26
