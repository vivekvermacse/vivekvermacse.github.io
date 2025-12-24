---
layout: post
title: "From Vectors to Meaning: Inside the Attention Mechanism"
date: 2023-03-15 10:00:00 -0800
categories: Deep Learning, Generative AI, Large Language Model, LLM
permalink: /2023/03/15/from-vectors-to-meaning-inside-the-attention-mechanism.html
---
<p><em>Updated: <time datetime="2025-03-20T10:00:00-08:00">March 20, 2025</time></em></p>

*The math that actually explains how transformers understand language*

If you have ever tried to dig into how Large Language Models (LLMs) like ChatGPT actually "read," you have almost certainly run into the word **Attention**.

Everyone says it is the secret sauce. But if you look for an explanation, you usually find one of two things: a hand-wavy analogy that oversimplifies it ("it just connects words together"), or a wall of academic jargon that drowns you in equations.

This essay takes a different approach.

We are going to start from human intuition—the way you are reading this sentence right now—and slowly peel back the layers until we reach the actual math running inside a Transformer. By the end, you won't just know *that* it works; you will be able to look at the attention equation and visualize exactly what every moving part is doing.

---

### Step 1: What “Attention” Means for Humans

Before we talk about vectors, let's talk about your brain. Consider this sentence:

> **"The cat sat on the mat because it was tired."**

When your eyes scan the word **"it"**, your brain doesn't process that word in isolation. If it did, "it" would be a meaningless placeholder. Instead, your brain instantly fires a connection back to **"cat"**, not "mat."

That mental linking is **attention**.

You are not treating all words equally. You are selectively focusing on the words that matter for the meaning you are currently trying to resolve. You ignore "the" and "on" because they aren't relevant to *who* is tired.

Humans do this constantly without thinking:

* **For pronouns:** it → cat
* **For cause-effect:** because → tired
* **For roles:** figuring out who did what to whom

Transformers do exactly the same thing—except they do it with vectors and matrix multiplication instead of biological neurons.

---

### Step 2: What Self-Attention Really Means

Transformers do not work with words; they work with math.

Every token (word or part of a word) is converted into an **embedding vector**—a list of numbers that captures meaning, grammar, and usage learned from reading terabytes of data.

**Self-attention** is simply the mechanism that allows these vectors to talk to each other. It asks: *When processing this specific word, how much should I care about every other word in the sentence?*

For our word **"it"**, the model might produce attention weights that look like this:

| Word | Attention Score |
| --- | --- |
| the | 0.04 |
| cat | 0.71 |
| sat | 0.05 |
| mat | 0.07 |
| because | 0.05 |
| tired | 0.03 |

Mathematically, this tells the model: *"When you update your internal understanding of 'it', borrow 71% of your new information from 'cat'."*

But where do these specific numbers come from? How does the model know that "it" matches "cat" and not "mat"?

---

### Step 3: The Mechanics of Communication (Q, K, and V)

This is where most explanations get confusing, so let's use a filing system analogy.

The model only has vectors. It doesn't know English grammar rules. It doesn't inherently know what a pronoun is. To solve this, we need a mathematical way for words to do three things:

1. **Ask** for what they need.
2. **Advertise** what they contain.
3. **Exchange** the actual information.

To achieve this, every single word vector is broken apart (projected) into three distinct roles:

1. **Query (Q):** What I am looking for?
2. **Key (K):** What do I represent? (How do I identify myself?)
3. **Value (V):** What information I actually contain

When we say **"it attends to cat,"** we are describing a specific interaction: The **Query** of "it" matches the **Key** of "cat." Because the lock fits the key, the **Value** of "cat" is passed over and injected into "it."

---

### Step 4: Where Q, K, and V Come From

Let's look at the actual math. Take our sentence:  
*"The cat sat on the mat because it was tired"*

Each word enters the layer as an embedding vector. For simplicity, let's imagine our vectors are just 4 dimensions long.

The model learns three massive weight matrices during training. These are the "knobs" the AI tunes to learn language.

To get the Query, Key, and Value for a word (let's call the word vector x), we simply multiply:

* Q = x · W_Q  
* K = x · W_K  
* V = x · W_V

Suddenly, "it" is no longer just one vector. It has split into a **Query vector** (looking for a noun), a **Key vector** (identifying itself as a pronoun), and a **Value vector** (its meaning).

---

### Step 5: Computing the Attention Score

Now, the "dating game" begins. We calculate how much "it" should attend to every other word by taking the dot product of the Query and the Keys.

**Hypothetical scores:**

* Score with cat: 3.5  
* Score with mat: 1.0  
* Score with sat: 0.8  

Normalize them using **Softmax**:

* cat: 0.70 (70%)  
* mat: 0.08 (8%)  
* sat: 0.05 (5%)

---

### Step 6: Where Value (V) Finally Matters

We have the weights, now we perform the retrieval.  

New representation for "it" is a weighted sum of the Value vectors:

* c_it = 0.71 · v_cat + 0.08 · v_mat + 0.05 · v_sat + …

The resulting vector is now dominated by the properties of "cat," enriched with context.

---

### Step 7: "But wait… aren't these weights random?"

At the start of training:

1. W_Q, W_K, W_V are random.  
2. Attention is random; "it" might attend to "mat" or "the."  
3. Model outputs garbage initially.

Training adjusts the matrices via **backpropagation**, nudging:

* Q to learn what pronouns should look for.  
* K to learn how nouns advertise themselves.  
* V to propagate the correct semantic meaning.

After billions of sentences, geometry emerges. Grammar is not programmed—it arises from the math.

---

### Step 8: Why Multi-Head Attention Exists

One attention mechanism isn't enough. Transformers use **multi-head attention**:

* Head 1: resolves pronouns  
* Head 2: links cause and effect  
* Head 3: tracks prepositions  

Outputs are concatenated for a rich, multi-dimensional understanding.

---

### Final Mental Model

A Transformer builds meaning by computing who should listen to whom—and how much.  

That is attention. It is not magic. It is not heuristics. It is geometry trained into language.

---

### References and Credits

1. **Vaswani et al.:** Attention Is All You Need  
   [Read the original paper here](https://arxiv.org/abs/1706.03762)

2. **Andrew Ng:** Deep Learning Specialization  
   [Check out the course here](https://www.deeplearning.ai/courses/deep-learning-specialization/)
   
3. **Grant Sanderson (3Blue1Brown):** Visualizing Attention
   [Watch the video here](https://www.youtube.com/watch?v=KJtZARuO3JY&t=2602s)
