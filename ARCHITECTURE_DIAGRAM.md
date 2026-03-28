# Architecture Diagram Guide

## Required Figure: architecture.png

Create a diagram showing the system architecture with the following components:

### Components to Include:

1. **User Input**
   - Text Input
   - Voice/Audio Input

2. **Speech-to-Text Module**
   - OpenAI Whisper Base
   - Audio Processing

3. **Intent Classification**
   - Fine-tuned DistilBERT
   - 8 Intent Classes

4. **Entity Extraction**
   - spaCy NER
   - Custom Pattern Matching
   - Entity Types: Names, Emails, Dates, Phones, Organizations

5. **Action Execution**
   - Google Calendar API
   - Gmail API
   - Session Management

6. **Response Generation**
   - Natural Language Response
   - Context Update

### Suggested Tools:

1. **draw.io** (Free, Online)
   - Go to https://app.diagrams.net/
   - Use flowchart templates
   - Export as PNG

2. **Lucidchart** (Free tier available)
   - Professional diagrams
   - IEEE-style templates

3. **TikZ** (LaTeX native)
   - Can be embedded directly in LaTeX
   - See example below

4. **PowerPoint/Keynote**
   - Simple shapes and arrows
   - Export as PNG

### Simple ASCII Reference:

```
┌─────────────────────────────────────────────────┐
│              USER INPUT                         │
│         (Text or Voice/Audio)                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         SPEECH-TO-TEXT (if audio)               │
│         OpenAI Whisper Base                     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         INTENT CLASSIFICATION                   │
│      Fine-tuned DistilBERT Classifier           │
│            (8 Intent Classes)                   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         ENTITY EXTRACTION                       │
│    spaCy NER + Custom Pattern Matching          │
│  (Names, Emails, Dates, Phones, Organizations)  │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         ACTION EXECUTION                        │
│  Schedule Meeting | Set Reminder | Send Email   │
│      Google Calendar | Gmail Integration        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         RESPONSE GENERATION                     │
│      + Session Memory Update                    │
└─────────────────────────────────────────────────┘
```

### TikZ Example (Alternative)

If you want to create the diagram directly in LaTeX, you can replace the `\includegraphics` command with TikZ code. Here's a simple example:

```latex
\begin{figure}[!t]
\centering
\begin{tikzpicture}[node distance=1.5cm, auto]
    \node [block] (input) {User Input};
    \node [block, below of=input] (stt) {Speech-to-Text};
    \node [block, below of=stt] (intent) {Intent Classification};
    \node [block, below of=intent] (entity) {Entity Extraction};
    \node [block, below of=entity] (action) {Action Execution};
    \node [block, below of=action] (response) {Response Generation};
    
    \path [line] (input) -- (stt);
    \path [line] (stt) -- (intent);
    \path [line] (intent) -- (entity);
    \path [line] (entity) -- (action);
    \path [line] (action) -- (response);
\end{tikzpicture}
\caption{System Architecture}
\label{fig:architecture}
\end{figure}
```

### Quick Creation Steps:

1. Use draw.io or similar tool
2. Create boxes for each component
3. Connect with arrows showing data flow
4. Use professional colors (blue/gray scheme)
5. Export as PNG at 300 DPI
6. Save as `architecture.png` in the same directory as `IEEE_Report.tex`

### Dimensions:

- Width: Should fit within one column (3.5 inches / 8.9 cm)
- Height: Flexible, but keep it reasonable (2-4 inches)
- Resolution: 300 DPI minimum for print quality

