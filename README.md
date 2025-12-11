# Memory Agent

Conversational agent with short-term and long-term memory management. 

## How to Use
1. Create virtual env
2. Install `requirements.txt`
3. While in `memory_agent/`, run `python main.py my_username` and each session use the same username!

## Happy Path!

When a user sends a message:
1. Context retrieval (parallel): Retrieves from short-term and long-term memory simultaneously
    -  Short-term: current topic messages (topics discussed more below) + all closed topic summaries from this session + full threads of highly relevant closed topics
    - Long-term: user profile facts, notepad, relevant topics from all past sessions (summaries only, NO full threads)
2. Response generation
3. In parallel with the generation is a fact extraction (user message gets passed into 4o-mini call to see if there are any facts to be gleaned for longterm)
4. Save exchange: Adds user/assistant messages to current open topic in short-term memory
5. Topic shift check: 
    - Only runs if current topic has >= 3 messages (design choice to put short, quick, topic changes together – e.g. question about ice cream, dinosaurs, and rainbows would be considered 1 topic)
    - More about topic shift...
        - Stage 1: Embedding similarity check (if similarity > 0.45, then consider it the same topic and skip step 4)
        - Stage 2: If similarity <= 0.45, LLM verifies if it's a new topic
    - If shift detected: closes old topic (generates summary in parallel), starts new topic with the turn that triggered shift. Some parallelization stuff to ensure that the summary must complete before next query's context is built (since the summary of prev. topic gets inserted in the next query's context!)
6. Fact saving: by this point, the fact extraction call from step 3 should have completed (since it runs concurrently with the actual response generation). Model here will score any extracted facts with importance scores as well. If they are >= 7 (out of 10), then it gets saves to long-term storage (`facts.json`)

At session end: Closes any open topic, updates notepad with strategic insights (another model call)

## Design decisions:
- I decided to try to create "topics" i.e. semantic groupings of conversation topics. Wanted to reach a few benefits from this:
    1. Short term context window: for really long conversation threads, rather than summarizing or keeping most recent-k messages (which loses detail about earlier messages in the thread), parts of conversation get grouped into "topics" whose summaries are embedded and can be retrieved depending on if they match the current thread of conversation. Allows for more detailed recall of previous conversation (if a topic gets retrieved for short-term, both the summary and the messages affiliated with that topic are injected as context)
    2. Easy object / structure for long term storage as well. The summaries that are generated for each topic + embeddings are stored in long term (not the messages themselves) which provide general guidance for how the user approached this conversation topic from last time. Also provides a more abstracted memory (that can be combined with the more strict, fact based memory) to form a more robust longterm memory.
- Separate retrieval for short-term vs long-term: short-term includes full message threads for relevant topics, long-term only includes summaries
- Merging of model calls / parallelization as often as possible. E.g:
    - Fact extraction parallelized with actual model response generation
    - Topic names generated in the same model call as summaries when topic closes (not during creation)
- Used pydantic for this to construct model output + implement a retry validator for if the structured output of the model doens't match! Did this because all providers / model SDKs have support for structured output
- Empty topics (0 messages) skipped from persistence, but topics with 1+ messages are persisted
- Separate embeddings from topics in LT storage for debugging ease (can look in and see topics, interpret them, etc.), but when these are loaded back into memory they are re-affiliated with their topics

## Notes on Short-Term Memory Strategy
I think this strategy is MAINLY advantageous for:
1. Really long contexts where you want to be able to retrieve detailed information from earlier parts of the conversation. 
2. Flexibly sized contexts. This is helpful if you are not sure what to set your "k" value at for a vanilla "show most recent k-messages" approach. In this "topic"-based strategy, the amount of detail you have varies based on if you switch conversation topics or not (i.e. if you have a long convo about the same topic then your context is long as well, but if you have a conversation where you switch topics often then your context is smaller as well.)

The main downside is there may be some information loss for very jumpy, detailed conversations. This is because once a topic switch is detected, the non-topic related messages get compressed into summaries and those summaries are fed instead of the raw messages themselves. Compared to a "keep most recent k messages" approach, this one may lose detail since it truncates / summarizes based on topic switch, not k-number of messages.

There is a way to implement this such that even after a topic switch, you continue providing non-topic related context to the model in direct message form. This would also be something fun to look into!

Other notes: 
- Added automatic closing for topics when context window limit reached (~25k chars)
- Includes summaries of all closed topics in the session
- For highly relevant closed topics (similarity > 0.75, max_k=2), includes full message threads

## Notes on Long-Term Memory Strategy
Persists user information and aggregates topics across all sessions.

Storage:
- `longterm_memory/{username}/facts.json` - user profile facts and observations. These are key:value pairs. Model is prompted to be STRICT about deciding what is a fact. Rated on importance and pruning / clearing of these facts, once the json gets full, is also based on importance rating.
- `longterm_memory/{username}/notepad.md` - scratchpad notes to allow for more freeform reflections. The model is prompted to "reflect" at the end of each session. Runs async so relatively lightweight!
- `longterm_memory/{username}/all_session_topics.json` - aggregated topics from all sessions. Normally, we may not want to save the full thread, but for debugging I've saved it so you can verify topic boundaries when you're testing!
- `all_session_topics.embeddings` - embeddings for cross-session retrieval

Session aggregation:
- At session end, any open topic is closed and all topics are added to `all_session_topics.json`
- Topic embeddings merged for semantic search across all sessions

Long term context construction:
- Always includes the user profile facts in KV form, and notepad
- Uses semantic search over `all_session_topics` to find relevant past conversations
- Returns top 3 most relevant topics (similarity > 0.5 threshold), injects their summaries as context (note that this differs from the retrieval in short term, where the top 2 ALSO have their full thread inserted to provide more detail.)
- Topic embeddings computed over summary text (or topic name if summary is placeholder)

## Tunable Parameters
These all can (and need) to be adjusted!

- MAX_CONTEXT_CHARS = 25000: Topic size limit before auto-closing a topic
- Topic shift detection: similarity_threshold = 0.45 (embedding similarity to skip LLM check)
    - This one is actually intentionally pretty high because you ideally want to be conservative with topic switches. Context windows are larger so you can afford to err on the side of having huge topics as opposed to many small topics (and risk losing information in the topic summarization / compression)
- Short-term retrieval: min_threshold = 0.75, max_k = 2 
- Long-term retrieval: min_threshold = 0.5, max_k = 3 
    - The thresholds for retrievals right now both may be too high. Oftentimes there are previous topics that may be good to include in the context that don't get detected.
    - This is probably also, more strongly, a problem with what the embeddings represent for each topic– the embedding is computed over the topic summary, which often is not a good semantic representation of the actual topic thread. E.g. the summaries will often start with "The user discussed...", so you might have a summary like: "The user discussed burger" and a query involving "The user interface" may have higher similarity scores than "Food", even though you'd want the latter query.


## Model Calls
Every user turn:
- Response generation: 1 call (main conversation)
- Fact extraction: 1 call (runs in parallel with response generation)
- Embedding calls: 2-3 calls (query embedding, context embedding for topic shift check, retrieval embeddings if topics exist)

On topic close:
- Topic summary generation: 1 call (runs in parallel with response generation if executor available)
- Embedding generation: 1 call (for closed topic summary/name)

On topic shift detection:
- LLM verification: 1 call (only if embedding similarity ≤ 0.45)

At session end:
- Notepad update: 1 call (only if session has topics with ≥2 messages)

Total per turn: Typically 3-4 LLM calls (response + facts + topic shift verification if needed), plus 2-3 embedding calls. Topic summarization adds 1 LLM call + 1 embedding call when topics close, but runs in parallel.

## Test Harness

Three test sessions to verify memory functionality. For an automated version of this, run:

`python eval/harness.py`

### Test 1: Session 1 - Short-Term Memory + Fact Storage

demonstrates: thread recall, pronoun resolution, fact extraction

```
user: Hi, my name is Alice and I'm a data scientist
user: I work with machine learning models at Google
user: I'm currently working on a project using TensorFlow
user: It's been challenging but really interesting
user: Can you help me understand how to optimize it?
```
end session: type `quit`

what to check:
- assistant should understand "it" refers to the TensorFlow project
- check `longterm_memory/{username}/facts.json` for stored facts with importance scores
- facts stored immediately after each turn
- after session ends...
    - topic w summary has been stored
    - potentially, some notes in notepad.md


### Test 2: Session 2 - Long-Term Memory Retrieval

demonstrates: fact recovery from session 1, topics from session 1

```
user: What's my name?
user: Where do I work?
user: Can you remind me about that TensorFlow project I was working on?
```

what to check:
- assistant recalls name and company from session 1 facts
- assistant retrieves TensorFlow topic from session 1
- check `longterm_memory/{username}/facts.json` - should have facts from session 1
- check `longterm_memory/{username}/all_session_topics.json` - should have TensorFlow topic
- context includes relevant past topics when querying about TensorFlow

end session: type `quit`

### Test 3: Session 3 - In-Session Topic Switching

demonstrates: detailed discussion, topic switching, maintaining context when switching back

```
user: I want to learn about neural networks and deep learning
user: Can you explain backpropagation in detail?
user: How does gradient descent work with backpropagation?
user: What are the differences between different activation functions?
user: Actually I want to go to Waffle House right now
user: How does their hash browns preparation work?
user: Can a person survive if they only eat Waffle House hashbrowns for 41 days without other nutrient intakes?
user: Going back to neural networks, what were the questions I had before that I was confused on?
```

what to check:
- assistant maintains detailed context about neural networks
- topic switch detected after "Waffle House" message (neural networks → food/restaurant)
- neural networks topic closed, waffle house topic started
- when switching back to neural networks, assistant recalls previous details (backpropagation, gradient descent, activation functions)
- check `sessions/{username}.{session_num}/topics.json` - should have closed neural networks topic
- check `sessions/{username}.{session_num}/topics.json` - should have waffle house topic (or current open topic)
- context retrieval finds neural networks topic when querying about previous questions (the answer should relate to what the summary of the neural networks topic contains!)
- session should finish with `closed topics in session: 1`

end session: type `quit`

### Where to Verify Results

**Facts Storage**
- `longterm_memory/{username}/facts.json` - all user facts with importance scores
- in-memory: `agent.memory.long_term.facts` - dict of facts

**Topics Storage**
- `sessions/{username}.{session_num}/topics.json` - closed topics in current session
- `longterm_memory/{username}/all_session_topics.json` - all topics across all sessions
- in-memory: `agent.memory.short_term.topics` - closed topics in current session
- in-memory: `agent.memory.long_term.all_session_topics` - all topics across sessions
