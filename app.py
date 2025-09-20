import streamlit as st
import json
import os
from news_generator import NewsGenerator
from similarity_calculator import SimilarityCalculator
from keyword_extractor import KeywordExtractor
import pandas as pd

# Configure page
st.set_page_config(
    page_title="News Brief Generator",
    page_icon="📰",
    layout="wide",
)

def main():
    st.title("📰 News Brief Generator")
    st.write("Generate AI-powered summaries of news articles with similarity evaluation")
    
    # Initialize components
    if 'generator' not in st.session_state:
        st.session_state.generator = NewsGenerator()
        st.session_state.similarity_calc = SimilarityCalculator()
        st.session_state.keyword_extractor = KeywordExtractor()
    
    # Model Selection
    st.header("🤖 AI Model Selection")
    
    # Get available models
    if 'generator' in st.session_state:
        available_models = st.session_state.generator.get_available_models()
        model_options = {}
        for model_id, info in available_models.items():
            model_options[f"{info['name']} ({info['speed']}, {info['context']} context)"] = model_id
        
        selected_model_display = st.selectbox(
            "Choose AI model for summary generation:",
            options=list(model_options.keys()),
            index=0,
            help="Different models offer various speeds and capabilities. Llama models are generally more capable, while Gemma is faster."
        )
        
        selected_model = model_options[selected_model_display]
        
        # Update model if changed
        if st.session_state.generator.model != selected_model:
            st.session_state.generator.set_model(selected_model)
            st.success(f"✅ Model updated to: {available_models[selected_model]['name']}")
    
    # Input section
    st.header("📝 Input Article")
    article_text = st.text_area(
        "Paste your news article here:",
        height=200,
        placeholder="Enter the news article text you want to summarize..."
    )
    
    # Similarity Scoring Configuration
    st.header("⚖️ Similarity Scoring Configuration")
    st.write("Adjust how similarity between original article and summaries is calculated:")
    
    col1, col2 = st.columns(2)
    with col1:
        cosine_weight = st.slider(
            "Cosine Similarity Weight (TF-IDF based)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values emphasize semantic content similarity based on word frequency and importance."
        )
    
    with col2:
        jaccard_weight = st.slider(
            "Jaccard Similarity Weight (Keyword overlap)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Higher values emphasize exact keyword matches between original and summary."
        )
    
    # Normalize weights to sum to 1.0
    total_weight = cosine_weight + jaccard_weight
    if total_weight > 0:
        normalized_cosine = cosine_weight / total_weight
        normalized_jaccard = jaccard_weight / total_weight
        st.info(f"🧮 Normalized weights: Cosine {normalized_cosine:.2f} + Jaccard {normalized_jaccard:.2f} = 1.00")
    else:
        st.warning("⚠️ Both weights are zero. Using default 0.7/0.3 ratio.")
        normalized_cosine = 0.7
        normalized_jaccard = 0.3
    
    if st.button("Generate Summaries", type="primary", disabled=not article_text.strip()):
        if article_text.strip():
            with st.spinner("Generating summaries and analyzing..."):
                try:
                    # Generate summaries
                    summaries = st.session_state.generator.generate_summaries(article_text)
                    
                    # Extract keywords and entities
                    original_keywords = st.session_state.keyword_extractor.extract_keywords(article_text)
                    bullet_keywords = st.session_state.keyword_extractor.extract_keywords(summaries['bullet'])
                    abstract_keywords = st.session_state.keyword_extractor.extract_keywords(summaries['abstract'])
                    simple_keywords = st.session_state.keyword_extractor.extract_keywords(summaries['simple'])
                    
                    # Extract named entities from original article
                    original_entities = st.session_state.keyword_extractor.extract_entities(article_text)
                    
                    # Calculate similarities with custom weights
                    bullet_similarity = st.session_state.similarity_calc.compute_similarity(
                        article_text, summaries['bullet'], original_keywords, bullet_keywords,
                        cosine_weight, jaccard_weight
                    )
                    abstract_similarity = st.session_state.similarity_calc.compute_similarity(
                        article_text, summaries['abstract'], original_keywords, abstract_keywords,
                        cosine_weight, jaccard_weight
                    )
                    simple_similarity = st.session_state.similarity_calc.compute_similarity(
                        article_text, summaries['simple'], original_keywords, simple_keywords,
                        cosine_weight, jaccard_weight
                    )
                    
                    # Determine best summary
                    similarities = {
                        'bullet': bullet_similarity,
                        'abstract': abstract_similarity,
                        'simple': simple_similarity
                    }
                    
                    best_summary_type = max(similarities.keys(), key=lambda k: similarities[k]['combined'])
                    
                    # Create results
                    results = {
                        "input_article": article_text,
                        "bullet_summary": summaries['bullet'],
                        "abstract_summary": summaries['abstract'],
                        "simple_summary": summaries['simple'],
                        "keywords": {
                            "original": original_keywords,
                            "bullet": bullet_keywords,
                            "abstract": abstract_keywords,
                            "simple": simple_keywords
                        },
                        "similarity": similarities,
                        "best_summary": best_summary_type,
                        "evaluation_table": create_evaluation_table(similarities),
                        "model_used": st.session_state.generator.model,
                        "similarity_weights": {
                            "cosine": normalized_cosine,
                            "jaccard": normalized_jaccard
                        },
                        "entities": original_entities
                    }
                    
                    # Store results in session state
                    st.session_state.results = results
                    
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating summaries: {str(e)}")
    
    # Display results if available
    if 'results' in st.session_state:
        display_results(st.session_state.results)

def create_evaluation_table(similarities):
    """Create markdown table for evaluation results"""
    table_data = []
    weights_info = None
    
    for summary_type, scores in similarities.items():
        table_data.append({
            'Summary Type': summary_type.title(),
            'Cosine Similarity': f"{scores['cosine']:.4f}",
            'Jaccard Similarity': f"{scores['jaccard']:.4f}",
            'Combined Score': f"{scores['combined']:.4f}"
        })
        
        # Get weights info from first entry (they're the same for all)
        if weights_info is None and 'weights' in scores:
            weights_info = scores['weights']
    
    df = pd.DataFrame(table_data)
    table_md = df.to_markdown(index=False)
    
    # Ensure table_md is not None
    if table_md is None:
        table_md = "Error generating evaluation table"
    
    # Add weights information if available
    if weights_info:
        table_md += f"\n\n**Scoring Weights:** Cosine {weights_info['cosine_weight']:.2f} + Jaccard {weights_info['jaccard_weight']:.2f}"
    
    return table_md

def format_entities_for_report(entities):
    """Format entities for text report"""
    if not entities:
        return "No named entities detected."
    
    # Group by type
    entities_by_type = {}
    for entity in entities:
        entity_type = entity['label']
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity['text'])
    
    # Format for report
    report_lines = []
    for entity_type, texts in entities_by_type.items():
        unique_texts = list(dict.fromkeys(texts))  # Remove duplicates
        report_lines.append(f"{entity_type}: {', '.join(unique_texts)}")
    
    return "\n".join(report_lines)

def display_results(results):
    """Display the analysis results"""
    st.header("📊 Results")
    
    # Best summary highlight
    best_type = results['best_summary']
    st.success(f"🏆 Best Summary: **{best_type.title()}** (Score: {results['similarity'][best_type]['combined']:.4f})")
    
    # Summaries section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔸 Bullet-Point Summary")
        st.write(results['bullet_summary'])
        st.caption(f"Keywords: {', '.join(results['keywords']['bullet'][:5])}")
    
    with col2:
        st.subheader("📄 Abstract Summary")
        st.write(results['abstract_summary'])
        st.caption(f"Keywords: {', '.join(results['keywords']['abstract'][:5])}")
    
    with col3:
        st.subheader("💬 Simple Summary")
        st.write(results['simple_summary'])
        st.caption(f"Keywords: {', '.join(results['keywords']['simple'][:5])}")
    
    # Evaluation table
    st.subheader("📈 Similarity Evaluation")
    st.markdown(results['evaluation_table'])
    
    # Add explanation of similarity metrics
    with st.expander("ℹ️ Understanding Similarity Metrics"):
        st.write("""
        **Cosine Similarity**: Measures semantic similarity based on word frequency and importance (TF-IDF vectors).
        Higher values indicate better content alignment.
        
        **Jaccard Similarity**: Measures exact keyword overlap between original article and summary.
        Higher values indicate more shared vocabulary.
        
        **Combined Score**: Weighted combination of both metrics using your configured weights.
        This balanced approach captures both semantic meaning and keyword preservation.
        """)
    
    # Keywords section
    st.subheader("🔑 Extracted Keywords")
    kw_col1, kw_col2 = st.columns(2)
    
    with kw_col1:
        st.write("**Original Article:**")
        st.write(", ".join(results['keywords']['original']))
    
    with kw_col2:
        st.write("**All Summary Keywords:**")
        all_summary_keywords = set()
        for kw_list in [results['keywords']['bullet'], results['keywords']['abstract'], results['keywords']['simple']]:
            all_summary_keywords.update(kw_list)
        st.write(", ".join(list(all_summary_keywords)))
    
    # Named Entities section (if available)
    if 'entities' in results and results['entities']:
        st.subheader("🏷️ Named Entities")
        
        # Group entities by type
        entities_by_type = {}
        for entity in results['entities']:
            entity_type = entity['label']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Display entities by type in columns
        if entities_by_type:
            entity_cols = st.columns(min(3, len(entities_by_type)))
            
            for idx, (entity_type, entities) in enumerate(entities_by_type.items()):
                with entity_cols[idx % len(entity_cols)]:
                    st.write(f"**{entity_type}** ({entities[0]['description']})")
                    entity_texts = [ent['text'] for ent in entities]
                    # Remove duplicates while preserving order
                    unique_entities = list(dict.fromkeys(entity_texts))
                    st.write(", ".join(unique_entities[:5]))  # Show top 5
                    if len(unique_entities) > 5:
                        st.caption(f"... and {len(unique_entities) - 5} more")
        
        # Expandable detailed view
        with st.expander("🔍 Detailed Entity Analysis"):
            for entity_type, entities in entities_by_type.items():
                st.write(f"**{entity_type}** - {entities[0]['description']}")
                entity_list = []
                for ent in entities:
                    entity_list.append(f"• {ent['text']}")
                st.write("\n".join(entity_list))
                st.write("")  # Add spacing
    else:
        st.info("ℹ️ Named entity recognition is enhanced with spaCy. If no entities appear, the article may not contain recognizable entities or spaCy may not be available.")
    
    # JSON output
    st.subheader("💾 JSON Output")
    with st.expander("View JSON Results"):
        st.json(results)
    
    # Download section
    st.subheader("⬇️ Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="📁 Download JSON",
            data=json_str,
            file_name="news_summary_results.json",
            mime="application/json"
        )
    
    with col2:
        # Create a formatted text report
        report = f"""News Brief Generator Report
===============================

Original Article:
{results['input_article'][:200]}...

SUMMARIES:
----------

Bullet-Point Summary:
{results['bullet_summary']}

Abstract Summary:
{results['abstract_summary']}

Simple Summary:
{results['simple_summary']}

EVALUATION:
-----------
{results['evaluation_table']}

Best Summary: {results['best_summary'].title()}

NAMED ENTITIES:
--------------
{format_entities_for_report(results.get('entities', []))}
"""
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="news_summary_report.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
