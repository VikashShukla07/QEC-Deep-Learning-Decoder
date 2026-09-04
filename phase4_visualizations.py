import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_model_evolution():
    # Data from our benchmarking
    models = ['Random Forest\n(Baseline)', 'PyTorch MLP\n(Weighted Loss)', 'PyTorch 2D-CNN\n(Spatiotemporal)']
    recall = [19.0, 93.47, 96.0]
    f1_scores = [0.32, 0.73, 0.79]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, recall, width, label='Recall (Catching Errors %)', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, [f * 100 for f in f1_scores], width, label='F1-Score (x100)', color='#4ECDC4')

    ax.set_ylabel('Percentage / Score')
    ax.set_title('QEC Decoder Evolution: Overcoming the Accuracy Paradox', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_evolution.png', dpi=300)
    print("Saved 'model_evolution.png'")
    plt.show()

def plot_final_confusion_matrix():
    # Final CNN Data from your terminal output
    cm = np.array([[17081, 897], 
                   [81, 1941]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Safe (Predicted)', 'Logical Flip (Predicted)'],
                yticklabels=['Safe (Actual)', 'Logical Flip (Actual)'],
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title('Final 2D-CNN Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Quantum State', fontsize=12)
    plt.xlabel('AI Predicted State', fontsize=12)
    plt.tight_layout()
    plt.savefig('cnn_confusion_matrix.png', dpi=300)
    print("Saved 'cnn_confusion_matrix.png'")
    plt.show()

if __name__ == "__main__":
    print("Generating Portfolio Visualizations...")
    plot_model_evolution()
    plot_final_confusion_matrix()