import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_triple_confusion_matrix():
    # The exact data from your previous terminal runs
    cm_rf = np.array([[17920, 58], 
                      [1630, 392]])
    
    cm_mlp = np.array([[16734, 1244], 
                       [132, 1890]])
    
    cm_cnn = np.array([[17081, 897], 
                       [81, 1941]])
    
    # Set up a 1x3 grid for the plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Evolution of QEC Decoder Performance (20,000 Test Shots)', fontsize=18, fontweight='bold', y=1.05)

    matrices = [cm_rf, cm_mlp, cm_cnn]
    titles = ['1. Random Forest (Baseline)\nProblem: High False Negatives (1,630)', 
              '2. PyTorch MLP (Weighted Loss)\nFix: High Recall, but High False Positives', 
              '3. PyTorch 2D-CNN (Spatiotemporal)\nResult: Balanced Precision & Recall']

    for i, ax in enumerate(axes):
        sns.heatmap(matrices[i], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Safe (Pred)', 'Flip (Pred)'],
                    yticklabels=['Safe (Actual)', 'Flip (Actual)'],
                    annot_kws={"size": 16, "weight": "bold"})
        ax.set_title(titles[i], fontsize=14, pad=15)
        ax.set_xlabel('AI Predicted State', fontsize=12)
        if i == 0:
            ax.set_ylabel('True Quantum State', fontsize=12)

    plt.tight_layout()
    plt.savefig('triple_confusion_matrix.png', bbox_inches='tight', dpi=300)
    print("✅ Saved 'triple_confusion_matrix.png'")
    plt.close()

if __name__ == "__main__":
    print("Generating Triple Matrix Comparison...")
    plot_triple_confusion_matrix()