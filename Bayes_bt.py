import pandas as pd
import numpy as np
import json
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_multiple_jsonl_files(file_paths):
    """Load multiple JSONL files and return list of DataFrames"""
    battles_list = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            battles_list.append(pd.DataFrame(data))
    return battles_list

def parse_score_label(score_str):
    """Parse score labels to numerical values"""
    if score_str is None:
        return 0.5
    
    score_map = {
        "A>>B": 1.0, "A>B": 0.75, "A=B": 0.5, "A<B": 0.25, "A<<B": 0.0,
        "B>>A": 0.0, "B>A": 0.25, "B=A": 0.5, "B<A": 0.75, "B<<A": 1.0
    }
    clean_score = score_str.strip("[]")
    return score_map.get(clean_score, 0.5)

def extract_comparison_scores(games_list):
    """Extract scores from games list"""
    scores = {}
    for game in games_list:
        comparison = game.get('comparison')
        score_str = game.get('score', '[[A=B]]')
        if comparison in ['baseline_vs_answer', 'answer_vs_baseline']:
            scores[comparison] = parse_score_label(score_str)
    return scores

def stage0_cross_file_aggregation(battles_list, aggregation_method='weighted_mean'):
    """Stage 0: Handle vertical inconsistency - aggregate across files"""
    all_battles = pd.concat(battles_list, ignore_index=True)
    
    grouped = all_battles.groupby(['uid', 'model', 'baseline', 'judge'])
    aggregated_results = []
    
    for name, group in grouped:
        if len(group) == 1:
            row = group.iloc[0].to_dict()
            row['repetition_count'] = 1
            row['vertical_uncertainty'] = 0.0
            aggregated_results.append(row)
        else:
            # Extract all scores for this comparison
            all_scores = {'baseline_vs_answer': [], 'answer_vs_baseline': []}
            valid_games = []
            
            for _, row in group.iterrows():
                scores = extract_comparison_scores(row['games'])
                for comp_type in all_scores:
                    if comp_type in scores:
                        all_scores[comp_type].append(scores[comp_type])
            
            # Aggregate scores
            aggregated_scores = {}
            uncertainties = {}
            
            for comp_type, score_list in all_scores.items():
                if score_list:
                    if aggregation_method == 'weighted_mean':
                        # Weight by inverse variance (higher weight for more consistent scores)
                        if len(score_list) > 1:
                            var = np.var(score_list) + 1e-6  # Add small constant to avoid division by zero
                            weights = [1/var] * len(score_list)
                        else:
                            weights = [1.0]
                        aggregated_scores[comp_type] = np.average(score_list, weights=weights)
                        uncertainties[comp_type] = np.std(score_list)
                    else:  # simple mean
                        aggregated_scores[comp_type] = np.mean(score_list)
                        uncertainties[comp_type] = np.std(score_list) if len(score_list) > 1 else 0.0
            
            # Construct aggregated games
            aggregated_games = []
            for comp_type, score in aggregated_scores.items():
                # Convert back to label format for consistency
                closest_label = min(["A>>B", "A>B", "A=B", "A<B", "A<<B"], 
                                   key=lambda x: abs(parse_score_label(f"[[{x}]]") - score))
                aggregated_games.append({
                    'round': 1,
                    'comparison': comp_type,
                    'score': f"[[{closest_label}]]"
                })
            
            avg_uncertainty = np.mean(list(uncertainties.values())) if uncertainties else 0.0
            
            aggregated_results.append({
                'uid': name[0],
                'model': name[1], 
                'baseline': name[2],
                'judge': name[3],
                'games': aggregated_games,
                'repetition_count': len(group),
                'vertical_uncertainty': avg_uncertainty
            })
    
    return pd.DataFrame(aggregated_results)

def get_original_labels(games_list):
    """Extract original labels from games list"""
    labels = {}
    for game in games_list:
        comparison = game.get('comparison')
        score_str = game.get('score', '[[A=B]]')
        if comparison in ['baseline_vs_answer', 'answer_vs_baseline']:
            labels[comparison] = score_str.strip("[]") if score_str else "A=B"
    return labels

################################################################################
def calculate_position_bias_agreement(label1, label2):
    """Calculate horizontal agreement based on label semantics"""
    # 无位置偏差: 方向完全对立且强度完全一致
    if ((label1 == "B>>A" and label2 == "A>>B") or 
        (label1 == "A>>B" and label2 == "B>>A") or
        (label1 == "B>A" and label2 == "A>B") or 
        (label1 == "A>B" and label2 == "B>A") or
        (label1 == "A=B" and label2 == "A=B")):
        return 1.0
    
    # 微弱位置偏差: 方向完全对立但强度有变化  
    elif ((label1 == "B>>A" and label2 == "A>B") or
          (label1 == "A>>B" and label2 == "B>A") or
          (label1 == "B>A" and label2 == "A>>B") or
          (label1 == "A>B" and label2 == "B>>A")):
        return 0.5
    
    # 显著位置偏差: 方向没有对立
    else:
        return 0.0

def stage1_position_bias_correction(battles_df):
    """Stage 1: Handle horizontal inconsistency - position bias correction"""
    corrected_data = []
    
    for _, row in battles_df.iterrows():
        games = row['games']
        if len(games) < 2:
            continue

        labels = get_original_labels(games)
        scores = extract_comparison_scores(games)
        
        baseline_vs_answer_label = labels.get('baseline_vs_answer')
        answer_vs_baseline_label = labels.get('answer_vs_baseline')
        baseline_vs_answer = scores.get('baseline_vs_answer')
        answer_vs_baseline = scores.get('answer_vs_baseline')
        
        if baseline_vs_answer is None or answer_vs_baseline is None:
            continue
        
        # Flip answer_vs_baseline for consistency
        answer_vs_baseline_flipped = 1.0 - answer_vs_baseline
        
        # Calculate horizontal agreement based on label semantics
        horizontal_agreement = calculate_position_bias_agreement(
            baseline_vs_answer_label, answer_vs_baseline_label
        )

        # Corrected score (position-bias adjusted)
        corrected_score = (baseline_vs_answer + answer_vs_baseline_flipped) / 2
        
        # Combined reliability weight (horizontal + vertical uncertainty)
        vertical_penalty = 1.0 / (1.0 + row.get('vertical_uncertainty', 0.0))
        horizontal_weight = 0.1 + 0.9 * horizontal_agreement
        combined_reliability = horizontal_weight * vertical_penalty
        
        corrected_data.append({
            'uid': row['uid'],
            'model': row['model'],
            'baseline': row['baseline'], 
            'judge': row['judge'],
            'corrected_score': corrected_score,
            'reliability_weight': combined_reliability,
            'horizontal_agreement': horizontal_agreement,
            'vertical_uncertainty': row.get('vertical_uncertainty', 0.0),
            'repetition_count': row.get('repetition_count', 1)
        })
    
    return pd.DataFrame(corrected_data)

def stage2_bradley_terry_modeling(corrected_df, min_comparisons=3):
    """Stage 2: Bradley-Terry mixed effects modeling"""
    
    # Filter models with sufficient comparisons
    model_counts = pd.concat([corrected_df['model'], corrected_df['baseline']]).value_counts()
    valid_models = model_counts[model_counts >= min_comparisons].index
    filtered_df = corrected_df[
        corrected_df['model'].isin(valid_models) & 
        corrected_df['baseline'].isin(valid_models)
    ].copy()
    
    if len(filtered_df) == 0:
        raise ValueError("Insufficient valid comparison data")
    
    # Encode models
    le = LabelEncoder()
    all_models = pd.concat([filtered_df['model'], filtered_df['baseline']]).unique()
    le.fit(all_models)
    
    # Prepare Bradley-Terry data format
    bt_data = []
    for _, row in filtered_df.iterrows():
        model_id = le.transform([row['model']])[0]
        baseline_id = le.transform([row['baseline']])[0]
        
        # Two rows per comparison (Bradley-Terry format)
        comparison_id = f"{row['uid']}_{row['judge']}"
        
        bt_data.append({
            'comparison_id': comparison_id,
            'model_id': model_id,
            'opponent_id': baseline_id, 
            'outcome': row['corrected_score'],
            'weight': row['reliability_weight'],
            'judge': row['judge']
        })
        
        bt_data.append({
            'comparison_id': comparison_id,
            'model_id': baseline_id,
            'opponent_id': model_id,
            'outcome': 1 - row['corrected_score'], 
            'weight': row['reliability_weight'],
            'judge': row['judge']
        })
    
    bt_df = pd.DataFrame(bt_data)
    
    # Fit mixed effects Bradley-Terry model
    try:
        # MixedLM doesn't support weights directly, so we'll replicate data based on weights
        weighted_bt_data = []
        for _, row in bt_df.iterrows():
            replications = max(1, int(row['weight'] * 10))  # Scale weights to replications
            for _ in range(replications):
                weighted_bt_data.append(row.to_dict())
        
        weighted_bt_df = pd.DataFrame(weighted_bt_data)
        
        model = mixedlm(
            "outcome ~ C(model_id)", 
            weighted_bt_df,
            groups=weighted_bt_df["comparison_id"]
        )
        result = model.fit(method='powell')
        
        # Extract model abilities
        model_abilities = {}
        for i, model_name in enumerate(le.classes_):
            if i == 0:
                model_abilities[model_name] = 0.0  # Reference model
            else:
                coef_name = f"C(model_id)[T.{i}]"
                model_abilities[model_name] = result.params.get(coef_name, 0.0)
        
        # Convert to DataFrame with winrates and confidence intervals
        abilities_df = pd.DataFrame([
            {'model': model, 'ability': ability} 
            for model, ability in model_abilities.items()
        ])
        
        abilities_df['winrate'] = 1 / (1 + np.exp(-abilities_df['ability']))
        
        # Calculate confidence intervals using parameter standard errors
        try:
            abilities_df['ability_se'] = [
                result.bse.get(f"C(model_id)[T.{i}]", 0.1) if i > 0 else 0.1 
                for i in range(len(le.classes_))
            ]
        except:
            abilities_df['ability_se'] = 0.1
            
        abilities_df['winrate_lower'] = 1 / (1 + np.exp(-(abilities_df['ability'] - 1.96 * abilities_df['ability_se'])))
        abilities_df['winrate_upper'] = 1 / (1 + np.exp(-(abilities_df['ability'] + 1.96 * abilities_df['ability_se'])))
        
        return abilities_df.sort_values('winrate', ascending=False), result
        
    except Exception as e:
        print(f"Mixed effects model failed: {e}")
        return simple_bt_fallback(filtered_df, le)

def simple_bt_fallback(corrected_df, le):
    """Simple Bradley-Terry fallback when mixed effects fails"""
    model_performance = {}
    
    for model in le.classes_:
        model_wins = corrected_df[corrected_df['model'] == model]['corrected_score']
        model_losses = corrected_df[corrected_df['baseline'] == model]
        
        total_score = model_wins.sum()
        if len(model_losses) > 0:
            total_score += (1 - model_losses['corrected_score']).sum()
            
        total_games = len(model_wins) + len(model_losses)
        model_performance[model] = total_score / max(total_games, 1)
    
    results_df = pd.DataFrame([
        {'model': model, 'winrate': performance}
        for model, performance in model_performance.items()
    ])
    
    results_df['winrate_lower'] = results_df['winrate'] - 0.05
    results_df['winrate_upper'] = results_df['winrate'] + 0.05
    
    return results_df.sort_values('winrate', ascending=False), None

def enhanced_hierarchical_evaluation(file_paths, aggregation_method='weighted_mean'):
    """Complete three-stage hierarchical evaluation pipeline"""
    
    print("=== Stage 0: Cross-file aggregation ===")
    battles_list = load_multiple_jsonl_files(file_paths)
    aggregated_df = stage0_cross_file_aggregation(battles_list, aggregation_method)
    print(f"Aggregated {sum(len(df) for df in battles_list)} evaluations into {len(aggregated_df)} samples")
    
    print("=== Stage 1: Position bias correction ===") 
    corrected_df = stage1_position_bias_correction(aggregated_df)
    print(f"Processed {len(corrected_df)} comparisons")
    print(f"Average horizontal agreement: {corrected_df['horizontal_agreement'].mean():.3f}")
    print(f"Average vertical uncertainty: {corrected_df['vertical_uncertainty'].mean():.3f}")
    
    print("=== Stage 2: Bradley-Terry modeling ===")
    leaderboard, model_result = stage2_bradley_terry_modeling(corrected_df)
    
    return leaderboard, corrected_df, model_result

def calculate_evaluation_metrics(old_leaderboard, new_leaderboard):
    """Calculate discrimination power improvement metrics"""
    var_old = np.var(old_leaderboard['winrate'])
    var_new = np.var(new_leaderboard['winrate'])
    
    ci_width_old = np.mean(old_leaderboard['winrate_upper'] - old_leaderboard['winrate_lower'])
    ci_width_new = np.mean(new_leaderboard['winrate_upper'] - new_leaderboard['winrate_lower'])
    
    return {
        'variance_improvement': (var_new - var_old) / var_old if var_old > 0 else 0,
        'ci_width_reduction': (ci_width_old - ci_width_new) / ci_width_old if ci_width_old > 0 else 0,
        'discrimination_power': var_new / ci_width_new if ci_width_new > 0 else 0
    }

if __name__ == "__main__":
    # Example usage
    file_paths = [
        ""
    ]
    
    try:
        leaderboard, corrected_data, model_result = enhanced_hierarchical_evaluation(
            file_paths, 
            aggregation_method='weighted_mean'
        )
        
        print("\n=== Final Leaderboard ===")
        print(leaderboard.round(3))
        
        print("\n=== Sample Corrected Data ===")
        print(corrected_data[['model', 'baseline', 'corrected_score', 'reliability_weight']].head(10))
        
    except Exception as e:
        print(f"Evaluation failed: {e}")


