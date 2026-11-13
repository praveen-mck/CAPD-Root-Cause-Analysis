--CHATS
-------------------------------------------------------------------------------------------------------------
--Input
-- DEV_MT_BIG_BETS_DB.POC.CHAT_TRANSCRIPTS_JULY_TO_SEPT_2025_V2
--Prediction Output
-- DEV_MT_BIG_BETS_DB.POC.CHAT_ANALYSIS_TAXONOMY_V9_CD
--Grading Output
-- DEV_MT_BIG_BETS_DB.POC.CHAT_ANALYSIS_TAXONOMY_V9_CD_GRADE
-------------------------------------------------------------------------------------------------------------  
-- All Results - Chats
CREATE OR REPLACE TABLE DEV_MT_BIG_BETS_DB.POC.RCA_ANALYSIS_PREDICTED_AND_GRADED_CHATS_v1
AS
SELECT
    inputt.body as CHAT_TRANSCRIPT,
    inputt.chat_transcript_name AS transcript_id,
    pred.*   EXCLUDE (chat_transcript_name, analyzed_at, is_no_input),
    grades.* EXCLUDE (chat_transcript_name, grader_run_id, graded_at)
FROM DEV_MT_BIG_BETS_DB.POC.CHAT_TRANSCRIPTS_JULY_TO_SEPT_2025_V2 AS inputt
JOIN DEV_MT_BIG_BETS_DB.POC.CHAT_ANALYSIS_TAXONOMY_V9_CD_GRADE AS grades
  ON inputt.chat_transcript_name = grades.chat_transcript_name
JOIN DEV_MT_BIG_BETS_DB.POC.CHAT_ANALYSIS_TAXONOMY_V9_CD AS pred
  ON inputt.chat_transcript_name = pred.chat_transcript_name
ORDER BY transcript_id;

select * from DEV_MT_BIG_BETS_DB.POC.RCA_ANALYSIS_PREDICTED_AND_GRADED_CHATS_v1


--CALLS
-------------------------------------------------------------------------------------------------------------
--Input
-- DEV_MT_BIG_BETS_DB.POC.RCA_CALLS_TRANSCRIPTS_JUN_TO_DEC_2025_V1
--Prediction Output
-- DEV_MT_BIG_BETS_DB.POC.CALLS_ANALYSIS_TAXONOMY_V9_CD_JUN_DEC_2025_PREDICTED
--Grading Output
-- DEV_MT_BIG_BETS_DB.POC.CALLS_ANALYSIS_TAXONOMY_V9_CD_JUN_DEC_2025_GRADED
-------------------------------------------------------------------------------------------------------------  
-- All Results - Chats
CREATE OR REPLACE TABLE DEV_MT_BIG_BETS_DB.POC.RCA_ANALYSIS_PREDICTED_AND_GRADED_CALLS_v1
AS
SELECT
    inputt.DIARIZED_TRANSCRIPT_TEXT as CALL_TRANSCRIPT,
    inputt.CALL_ID AS transcript_id,
    pred.*   EXCLUDE (CALL_ID, analyzed_at, is_no_input),
    grades.* EXCLUDE (CALL_ID, grader_run_id, graded_at)
FROM DEV_MT_BIG_BETS_DB.POC.RCA_CALLS_TRANSCRIPTS_JUN_TO_DEC_2025_V1 AS inputt
JOIN DEV_MT_BIG_BETS_DB.POC.CALLS_ANALYSIS_TAXONOMY_V9_CD_JUN_DEC_2025_GRADED AS grades
  ON inputt.CALL_ID = grades.CALL_ID
JOIN DEV_MT_BIG_BETS_DB.POC.CALLS_ANALYSIS_TAXONOMY_V9_CD_JUN_DEC_2025_PREDICTED AS pred
  ON inputt.CALL_ID = pred.CALL_ID
ORDER BY inputt.CALL_ID;

select * from DEV_MT_BIG_BETS_DB.POC.RCA_ANALYSIS_PREDICTED_AND_GRADED_CALLS_v1

