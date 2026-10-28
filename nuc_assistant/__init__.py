"""
NUC Offline Voice Assistant
Intel NUC i5-1135G7 | 32GB RAM | Intel Iris Xe

Pipeline:
  openWakeWord → Silero VAD + Smart Turn → faster-whisper
    → Ollama/Gemma (streamed) → Piper TTS → Speaker

Modules:
  config    load and check config.yaml
  audio     microphone recording and speaker playback
  listen    voice activity, end of turn (Smart Turn) and barge-in
  wakeword  wake word detection (openWakeWord)
  stt       speech-to-text (faster-whisper)
  llm       chat with the LLM (Ollama), streamed sentence by sentence
  tts       text-to-speech (Piper)
  main      the main conversation loop
"""
