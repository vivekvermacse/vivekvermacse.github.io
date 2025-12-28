---
layout: post
title: "Beyond Attention: How Language Models Truly Think"
date: 2023-04-12 10:00:00 -0800
categories: Deep Learning, Generative AI, Large Language Model, LLM
permalink: /2023/04/12/beyond-attention-how-language-models-truly-think.html
---

<p><em>Updated: <time datetime="2025-03-31T10:00:00-08:00">March 31, 2025</time></em></p>

*Tokenization, positional encoding, and the MLP — the quiet engines behind the magic*

Let’s be honest: in the world of LLMs, Attention steals the show.

If you ask anyone how ChatGPT works, they usually point to the "Attention Mechanism." And they aren't wrong—Attention is brilliant. It’s the spotlight that lets the model scan a sentence and realize that the word "it" is actually talking about "cat." in the sentence "The cat sat on the mat because it was tired."

To understand how a model actually thinks, imagine an investigative journalist working on a story:

Tokenization & Embeddings are the Field Reporters. They take the messy, raw events of the world and translate them into a coded language of facts and data points that the newsroom can process.

Positional Encoding is the Time-Stamping. Before any analysis happens, every piece of data is marked with exactly when and where it occurred. Without this, the journalist wouldn't know if the "crime" happened before or after the "arrest."

Attention is the Connection-Seeker. This is the journalist pinning photos to a corkboard and drawing red strings between them. It realizes, "Hey, this person (it) is actually linked to this event (the cat)."

The MLP (Feed-Forward Network) is the Editor. After the strings are drawn, the Editor sits down to actually write the story. They take those raw connections and turn them into a logical narrative. They are the "brains" that decide what the connections actually mean.

Without the reporters and the time-stamps, the journalist has no data to look at. And without the Editor, all you have is a corkboard full of string but no actual story.

To process a sentence like “The cat sat on the mat because it was tired,” these heroes have to work in a specific, rhythmic sequence. They are the reason an LLM feels like a reasoning engine rather than just a sophisticated version of autocomplete.

Let’s meet the unsung heroes of the LLM world.

---

### 1. Tokenization

**How raw text becomes something the model can understand**

A Transformer doesn’t see letters or words like we do. It doesn’t know grammar or spelling. To the model, a sentence is just a sequence of numbers.

Take a simple sentence:

> “The cat sat on the mat.”

The first step is **tokenization** — breaking text into **tokens**, the smallest pieces the model can work with. Tokens can be full words, parts of words, punctuation marks, emojis, or other symbols.

In the early days of NLP, every word got its own number. For example, “cat” = 4932, “dog” = 4821, and so on. But this approach has limits: what happens with unusual words, slang, or completely new words like *“uninstagrammable”*? They would break the model.

Modern LLMs handle this elegantly using **tokenizers** that can split words and symbols into manageable pieces.

- Common words often remain whole: `apple` → `apple`  
- Rare or long words are split: `unbelievable` → `un` + `believ` + `able`  
- Emojis and punctuation become their own tokens: `😀`, `.`, `?`  

Each token is then **mapped to a token ID** using a fixed-size dictionary called the **vocabulary**. Vocabulary size is chosen when the model is created, typically ranging from tens of thousands to hundreds of thousands of tokens depending on the model.

**Examples of modern tokenization systems:**

- **BPE (Byte-Pair Encoding)** — GPT-2/3  
- **WordPiece** — BERT  
- **SentencePiece / Unigram** — T5, LLaMA  

So our sentence becomes a sequence of **token IDs**:

| Token | ID   |
|------|------|
| The  | 1823 |
| cat  | 4932 |
| sat  | 912  |
| on   | 271  |
| mat  | 4418 |

These token IDs are often **converted into one-hot vectors**. A one-hot vector is mostly zeros, except for a single 1 at the index corresponding to the token ID. For example, if the vocabulary size is 50,000 and the token ID is 4932, the vector will have a 1 at position 4932 and 0 everywhere else.

This step might seem simple, but it’s crucial. It’s how the model turns messy human language into something it can process mathematically. Without tokenization and this mapping, nothing else in the model could work.

---

### 2. Turning IDs into Meaning: Embeddings

**How numbers start to represent words**

At this point, we have a sequence of **token IDs**—numbers like `[1823, 4932, 912, ...]`. But these numbers are just labels. The ID `4932` doesn’t know anything about a cat. To the model, `4932` is no closer to `dog` than it is to `microwave`.

We need a way to turn these token IDs into **mathematical vectors** that carry meaning. This is where the **embedding layer** comes in.

---

#### What an Embedding Is

Think of the embedding layer as a giant dictionary stored as a **matrix (E)**. Each row corresponds to a token ID and contains a dense vector of numbers—its embedding. For example:

- Token `cat` → row 4932 in (E) → `[0.12, -0.54, 0.33, ...]`

The **dimension** of these vectors (often 768, 1024, or higher) is a design choice. Higher dimensions allow more nuanced meanings but increase computation and memory usage.

---

#### Learning vs. Contextualization

Here’s the key nuance:

1. **During training**, the embedding matrix (E) is **learned together with the rest of the model**. It starts as random numbers, and gradient updates gradually adjust each token’s embedding so the model can best understand language. This makes (E) optimized for the **entire LLM**, rather than just each word in isolation.

2. **After training**, (E) is **frozen**. It becomes a static mapping: each token ID has exactly one row in (E).

3. **Contextual meaning comes from the rest of the model**—the **Multi-Head Attention** and **MLP layers**. These layers mix and process the embeddings depending on surrounding words. That’s why the same token ID `bank` can produce very different contextual embeddings in:

> “I sat by the **bank** of the river.”  
> “The **bank** is closed today.”

Even though the row for `bank` in (E) is static, the final representation that goes into later layers is **dynamic and context-aware**.

You could technically replace (E) with pre-trained static embeddings (like Word2Vec or GloVe), but the overall model performance is usually worse because the embeddings are **not tuned jointly** with the attention and MLP layers. Learning (E) during training ensures the entire LLM is optimized end-to-end.

---

#### From Token ID to One-Hot to Embedding

The flow looks like this:

1. **Token ID**: `4932`
2. **One-hot vector**: mostly zeros, with a 1 at position 4932
3. **Embedding lookup**: multiply the one-hot vector by the embedding matrix (E) to produce a dense vector

**Embedding = OneHot(ID) · E**

Modern frameworks optimize this without creating huge sparse vectors, but conceptually this is how the mapping works.

---

#### Real-World Examples

- **GPT-3**: Vocabulary ~50,000 tokens, embedding dimension 1,280  
- **LLaMA-2**: Vocabulary 32,000 tokens, embedding dimension 4,096  
- **PaLM 2**: Vocabulary ~1 million tokens, embedding dimension 2,560  

Each token gets a high-dimensional vector that captures semantic, grammatical, and syntactic information, ready for the attention layers to contextualize.

---

#### Key Takeaways

- Embeddings turn meaningless numbers into vectors that carry information.  
- The learned embedding matrix (E) is **static after training**, but the **contextual embeddings vary** based on surrounding tokens.  
- End-to-end learning of (E) with attention and MLP layers produces the **best performance**.  
- Replacing (E) with pre-trained static embeddings is possible but usually suboptimal.

---

### 3. Positional Encoding: Giving Words a Sense of Order

**Why "The dog bites the man" isn’t the same as "The man bites the dog"**

By now, we have **token embeddings** for each word. But there’s a catch: Transformers **process all tokens in parallel**. That makes them super fast, but also **order-blind**.

Imagine feeding the sentence:

1. *"The dog bites the man"*
2. *"The man bites the dog"*

If we only gave the embeddings to the model, it would see both sentences as the same set of vectors—just shuffled around. The model wouldn’t know which word came first.

To fix this, we need to **inject a sense of position** into each token.

---

#### How Position is Added

Transformers do this with **positional encoding**, a clever mathematical trick that assigns a **unique vector to each position** and adds it to the token embedding.

The classic approach uses **sine and cosine functions** of different frequencies. For position (`pos`) and embedding dimension (`i`):

**PE(pos, 2i) = sin( pos / 10000^(2i / d_model) )**  
**PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )**

Where:

- `d_model` = embedding size  
- `i` = dimension index  
- Even dimensions use **sine**, odd dimensions use **cosine**

Then the positional encoding is **added element-wise** to the token embedding.

---

#### Why Sine and Cosine Works

1. **Unique for each position**: Each position produces a vector different from all others, because sine and cosine values vary smoothly with position.  
2. **Smooth gradients**: The functions are continuous, so the model can learn relationships between nearby positions (helpful for relative distances).  
3. **Multiple frequencies**: Using different frequencies across dimensions ensures that long sequences are uniquely encoded and positions are distinguishable.

**Example (toy version)**:

Suppose embedding dimension \(d_{\text{model}} = 4\). For positions 1 and 2:

\[
\begin{aligned}
\text{PE}_{pos=1} &= [\sin(1/10000^{0/4}), \cos(1/10000^{0/4}), \sin(1/10000^{2/4}), \cos(1/10000^{2/4})] \\
&\approx [0.8415, 0.5403, 0.01, 0.99995] \\
\text{PE}_{pos=2} &= [\sin(2/10000^{0/4}), \cos(2/10000^{0/4}), \sin(2/10000^{2/4}), \cos(2/10000^{2/4})] \\
&\approx [0.9093, -0.4161, 0.02, 0.9998]
\end{aligned}
\]

- Notice how each position vector is **distinct**.  
- Adding this to the token embedding gives the model **both meaning and location** for every word.

---

#### Real-World Notes

- Maximum sequence lengths: GPT-3 uses 2,048 tokens, LLaMA-2 up to 4,096, and Claude 3 can go even higher.  
- Positional encoding vectors are **added** to embeddings before any attention layers, so each word “knows where it sits” in the sentence.  
- This is especially important for long sequences, because attention only sees the vectors, not the original word order.

---

#### Mental Picture

Think of the token embeddings as **colors**, and the positional encoding as a **tint of location**. When you mix them, every word has both its semantic color and its position in the sequence. Attention can then use this to figure out relationships like:

> "The subject usually comes before the verb"  
> "Modifiers refer to the closest noun"

Without positional encoding, the model would lose this structure.

---

### 4. Multi-Head Attention — Looking at the Sentence from Many Angles

Now that each word has both meaning (from embeddings) and position (from positional encoding), it’s time for the model’s first major act of reasoning: **Multi-Head Attention**.

As explained in [Part 1 of this series](https://medium.com/@vivekverma.cse/from-vectors-to-meaning-inside-the-attention-mechanism-a2fabbb2e1f4), Attention lets each word ask, *“Who should I listen to in this sentence?”* It computes similarity scores and blends information across all tokens. But real language is too rich to be captured with just one perspective at a time.

That’s why Transformers use **multiple attention heads in parallel**.

Each head is like a different pair of glasses:

- One head might specialize in resolving pronouns (e.g., *“it” → “cat”*).  
- Another head might focus on adjectival relations (e.g., *“big” → “ball”*).  
- Yet another might discover syntactic dependencies or negation patterns.  

All heads compute attention independently. Their outputs are then concatenated and linearly projected so the model gets a richer, more nuanced representation before it moves on to the next layer.

Here’s the key conceptual takeaway:

> **Multi-Head Attention is like looking at the same sentence through several different semantic and syntactic lenses at the same time.**  
> Each lens highlights a different pattern, and the model learns how to combine them.

You don’t need to memorize the math here; just remember that multiple heads give the model **breadth of understanding**, not just depth. For a full, intuitive walkthrough of how Query, Key, and Value work in self-attention, see my detailed explanation in  
[Part 1 — *From Vectors to Meaning: Inside the Attention Mechanism*](https://medium.com/@vivekverma.cse/from-vectors-to-meaning-inside-the-attention-mechanism-a2fabbb2e1f4).

---
### **5. The Brain: Feed-Forward Network (MLP)**

After Multi-Head Attention has blended information from all words, we reach the stage where the model actually starts **“processing” meaning**. This is the **MLP (Multi-Layer Perceptron)**, also called the **Feed-Forward Network**, inside each Transformer block.

Think of Attention as gathering all the ingredients from the kitchen. It knows which onions, tomatoes, or garlic to pick, and in what proportion. But **it doesn’t cook anything**. The MLP is the chef that takes these ingredients and transforms them into a finished dish.

Here’s what happens in the MLP:

1. **Expansion:** The input vector from Attention is projected into a much larger space, often 4× the original dimension. This gives the network more “room” to mix and explore patterns.  
2. **Non-Linearity:** An activation function (like ReLU or GeLU) is applied. This step is critical — it allows the network to learn **complex, non-linear relationships** between words. Without it, the entire model would collapse into a simple linear system that cannot represent meaning.  
3. **Contraction:** Finally, the vector is projected back to the original size so it can be passed to the next layer.

If you want a deeper intuition for why this non-linear step is so powerful, I’ve explained it in detail here:  
[Why Non-Linear Activation Functions Make Deep Learning Possible](https://medium.com/@vivekverma.cse/why-non-linear-activation-functions-make-deep-learning-possible-a492c48e0d47)

At the end of this process, the vector for each token isn’t just a mix of other words; it’s now **contextually enriched**. For example:

- The word **“bank”** in *“The bank of the river is wide”* will have a different vector than in *“The bank is closed for a holiday.”*  
- Even though the **Embedding Matrix E** gives “bank” a single initial representation, the combination of **Multi-Head Attention + MLP** produces **dynamic, context-dependent embeddings**.

A few real-world notes:

- The embedding dimension (`d_model`) is often in the **thousands** (e.g., 768 for BERT-base, 12,288 for GPT-4).  
- The MLP expansion might take that 768-dimensional vector up to **3,072** dimensions before projecting it back.  
- These layers are **fully learned during training**, allowing embeddings, attention, and MLP weights to co-evolve into a highly optimized system.

So, in a nutshell:

> **Attention routes the information. The MLP digests it and adds depth.**

Without this stage, the model would only shuffle information around without truly **understanding** context, relationships, or subtle patterns in the sentence.

---
### **6. Predicting the Next Word**

After passing through dozens of Transformer layers — each with **Multi-Head Attention** mixing context and **MLPs** enriching meaning — we arrive at the final vector for the current token. This vector now carries a **rich, context-dependent representation** of the word in the sentence.

But how does the model pick the **next word**?

1. **Un-embedding:** We multiply the final token vector by the **transpose of the embedding matrix** (E^T). This projects the vector back into the vocabulary space, giving a raw score for every possible token.
2. **Softmax:** These raw scores are converted into probabilities using the **Softmax function**, creating a distribution over the entire vocabulary.

For example, given the input:

> *“The cat sat on the”*

The model might output something like:

| Token     | Probability |
| --------- | ----------- |
| **mat**   | 0.85        |
| **sofa**  | 0.05        |
| **floor** | 0.07        |
| **bank**  | 0.03        |

The model picks **“mat”**, and the process repeats for the next token. Over time, this autoregressive generation can produce entire paragraphs of coherent text.

---

### **Final Mental Model: How an LLM Really Thinks**

It’s tempting to reduce LLMs to just **“Attention machines.”** But that misses the symphony of components working together. Here’s the full picture:

1. **Tokenization:** Breaks raw text into manageable pieces (tokens), including punctuation, emojis, or subword units. Each token is mapped to an ID via a fixed dictionary and optionally converted to a one-hot vector.
2. **Embeddings:** Turns IDs into dense vectors. These embeddings are **initially static**, but the combination of Attention + MLP produces **dynamic, contextual representations** during inference.
3. **Positional Encoding:** Adds location information so the model knows the order of tokens. Sinusoidal encodings guarantee each position is uniquely identifiable.
4. **Multi-Head Attention:** Allows each token to “look at” other tokens and mix information efficiently. Check [Part 1](https://medium.com/@vivekverma.cse/from-vectors-to-meaning-inside-the-attention-mechanism-a2fabbb2e1f4) for a deep dive into the math.
5. **MLP / Feed-Forward Network:** Digests the mixed information, adding depth, non-linearity, and reasoning to each token vector. See [Why Non-Linear Activation Functions Make Deep Learning Possible](https://medium.com/@vivekverma.cse/why-non-linear-activation-functions-make-deep-learning-possible-a492c48e0d47) for more context.
6. **Output Layer:** Projects the processed token vector back to vocabulary space and predicts the next word using Softmax.

Some real-world numbers for context:

* GPT-3 (175B) uses **2048–4096 token context windows**, split into chunks during training. Each chunk is processed independently, but the model learns patterns over long contexts.
* Embedding dimension (`d_model`) is in the **thousands**, allowing rich, nuanced representations.
* Chunking and batching make it feasible to train on **hundreds of gigabytes to terabytes of text**, while the network learns **all weights jointly**, optimizing embeddings, attention, and MLP layers for the best performance.

---

**Key Takeaways**

* **Attention** is like a traffic controller — it routes information efficiently.
* **MLP** is the processor — it transforms context into meaningful representations.
* **Embeddings** are the dictionary — initially static, later dynamically enriched by context.
* **Positional Encoding** ensures that order matters in language.

---

Together, these hidden heroes turn **raw internet text** into something that behaves like understanding. They may be behind the scenes, but without them, Attention alone would be powerless — just shuffling meaningless noise.

