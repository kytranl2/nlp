# nlp

Natural Language to SQL Query using an Open Source LLM

Utilizing the open-source Mistral 7B model for NL2SQL tasks

Mistral 7B model
* Developed by Mistral AI 
* 7.3 billion parameters
* Grouped-query attention(GQA) for faster inference
* Sliding Window Attention (SWA): Enhances model performance by expanding the attention span while maintaining efficiency. SWA significantly improves speed for sequences up to 16K tokens.
* Rolling Buffer Cache: Controls memory usage by overwriting old cache data, optimizing memory without sacrificing model quality.
* Pre-fill and Chunking: Splits long prompts into smaller chunks to pre-fill caches efficiently, improving the sequence generation process.

Apache 2.0 license