You are helping me prepare to facilitate a hackathon challenge. I am a Forward Deployed Engineer (FDE) at the AI Engineering Lab. Tomorrow I will be the technical anchor for a team of 3–5 government engineers building a working prototype in one day.

I have not read the challenge brief yet. I need you to make me the expert in the room. Produce a comprehensive **Facilitator Preparation Guide** covering every section listed below. Be thorough, practical, and opinionated — I need to unblock teams, steer them away from dead ends, and help them produce a strong demo by 15:00.

## Context

- **Challenge I am facilitating:** [REPLACE WITH ONE OF: "Challenge 1: From PDF to digital service" | "Challenge 2: Unlocking the dark data" | "Challenge 3: Supporting casework decisions" | "Challenge 4: Knowing your own organisation"]
- **My preferred AI coding tool:** [REPLACE WITH ONE OF: "GitHub Copilot" | "Claude Code" | "Amazon Kiro" | "Other"]

The hackathon is run by the AI Engineering Lab (part of DSIT — UK Government). Teams are civil servant engineers of mixed experience levels. They use AI coding tools throughout the day to plan, build, and test. The event does NOT provide access to AI model APIs for use inside the application — teams can mock AI endpoints, use their own model access, or build without AI in the app. Judging combines milestone points earned during the day with a rubric-based review at each table. There are no stage presentations.

## Sections to cover

### 1. Challenge deep-dive
- Summarise the problem in 2–3 paragraphs as if explaining it to a colleague who knows nothing about it.
- Who are the users? What does their day look like today? What is painful?
- Why does this matter to government at scale?
- What does a strong prototype look like by end of day — be specific about the minimum viable demo.

### 2. Domain knowledge I need
- Key government concepts, terminology, or policy patterns relevant to this challenge that I should understand before the day.
- Any GOV.UK design patterns, service standards, or accessibility requirements that are relevant.
- If data is provided, walk me through the data model — what fields exist, how records relate to each other, what patterns or deliberate anomalies are embedded in the starter data.

### 3. Architecture and technology recommendations
- Recommend 2–3 realistic tech stack options for a one-day prototype (e.g. plain HTML/CSS/JS, React, Python Flask, GOV.UK Prototype Kit). For each, state who it suits and the tradeoff.
- Describe the simplest viable architecture — what components are needed and how they connect.
- If the challenge involves AI capabilities: explain clearly what can be mocked, what requires a real model, and how to architect the boundary so mocks can be swapped for real calls later.

### 4. Common pitfalls and dead ends
- List the top 5–7 mistakes teams typically make on this type of challenge. For each, explain the warning sign I should watch for and what I should say to redirect them.
- Include both technical pitfalls (e.g. spending too long on auth, over-engineering the database) and scoping pitfalls (e.g. trying to cover too many user journeys, polishing UI before the core flow works).

### 5. Time-boxed milestone plan
- Provide a detailed hour-by-hour plan from 09:15 (problem selection) to 15:00 (build phase closes), mapped to the hackathon schedule.
- For each time block, specify: what the team should have completed, what I should check, and what to do if they are behind.
- Include explicit "scope cut" decision points — moments where I should push the team to drop features and focus on a working demo.

### 6. AI coding tool strategy (for my specific tool)
- Give me 10–15 specific, copy-paste-ready prompts I can suggest to my team at different stages of the day (planning, scaffolding, building features, writing tests, preparing for the judge review).
- For each prompt, explain when to use it and what good output looks like.
- Include guidance on how to get the best results from my specific tool — what it is strong at, what it struggles with, and how to recover when it gives poor output.
- Include prompts for: scoping the problem, generating starter code, implementing validation, writing tests, accessibility auditing, and preparing a 2-minute judge explanation.

### 7. Unblocking playbook
- For each of the following common blockers, give me a diagnosis approach and a specific fix:
  - Team cannot agree on what to build
  - First prototype attempt does not work and team wants to start over
  - Team is stuck on a specific technical problem (data parsing, API integration, styling)
  - Team is building too much and will not have a working demo
  - Team has a working demo early and is not sure what to do next
  - One team member is disengaged or blocked
  - Team has no access to AI model APIs and the challenge seems to need one

### 8. Judging preparation
- What questions will the judges ask? List the likely questions and coach me on what a strong answer sounds like versus a weak one.
- What should the team be able to demonstrate in their 5–10 minute judge visit?
- What should they be honest about (what is mocked, what they would do next, what failed)?
- Give me 3 prompts the team can run through their AI tool in the final 30 minutes to prepare for the judge visit.

### 9. Stretch goals and "wow" factors
- If the team finishes the core prototype early, suggest 3–5 stretch goals ranked by impact-to-effort ratio.
- What would make a judge remember this team's demo over the others?
- Any contrarian or unexpected directions that would be technically impressive and still achievable?

### 10. Quick reference card
- Produce a single-page summary I can keep open on my laptop all day: key URLs, data file locations, the 5 most important prompts, the 3 biggest pitfalls, and the milestone checkpoints.

## Formatting
- Use clear headers and subheaders.
- Use tables where they aid scanning.
- Keep copy-paste prompts in fenced code blocks.
- Be direct and opinionated — I want your best judgement, not a menu of options.
- Use British English spelling throughout.