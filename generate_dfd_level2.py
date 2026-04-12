import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(18, 12))

# Styles
ext_entity = dict(boxstyle="square,pad=1.0", fc="#f1f5f9", ec="#64748b", lw=2)
process = dict(boxstyle="round,pad=1.2", fc="#eff6ff", ec="#2563eb", lw=2)
datastore = dict(boxstyle="round4,pad=1.0", fc="#fff7ed", ec="#ea580c", lw=2)

# Nodes mapping with explicit calculated absolute coordinates
nodes = {
    'Input': (0.1, 0.55, "1.0 Frame\nPre-Processing", ext_entity, "#0f172a"),
    '2.1': (0.3, 0.55, "2.1 MediaPipe\nFace Mesh\nExtraction", process, "#0f172a"),
    
    '2.2': (0.54, 0.82, "2.2 Calculate\nEye Aspect\nRatio (EAR)", process, "#0f172a"),
    '2.3': (0.54, 0.64, "2.3 Calculate\nYaw & Pitch\n(Head Pose)", process, "#0f172a"),
    '2.4': (0.54, 0.46, "2.4 MobileNet\nCNN Image\nClassification", process, "#0f172a"),
    '2.5': (0.54, 0.28, "2.5 dlib 128D\nFace Embedding\nGeneration", process, "#0f172a"),
    
    'DB': (0.54, 0.08, "D1 Student\nSQLite Database", datastore, "#0f172a"),
    
    '2.6': (0.81, 0.55, "2.6 Hybrid Fusion\n& Rolling Avg\nAggregation", process, "#0f172a"),
    'Output': (1.0, 0.55, "3.0 Dashboard\nWebsocket API", ext_entity, "#0f172a") 
}

# Draw Nodes
for k, v in nodes.items():
    ax.text(v[0], v[1], v[2], ha="center", va="center", size=11, bbox=v[3], color=v[4], weight='bold', zorder=5)

def draw_arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2.5), zorder=2)

def draw_label(x, y, text, fc="white", ec="none"):
    ax.text(x, y, text, ha="center", va="center", size=10, weight="bold", color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.2", fc=fc, ec=ec, lw=1, alpha=0.9), zorder=6)

# Input -> 2.1
draw_arrow(0.165, 0.55, 0.21, 0.55)
draw_label(0.187, 0.58, "RGB Video\nFrames")

# 2.1 -> Merges 
ax.plot([0.39, 0.42], [0.55, 0.55], color="#475569", lw=2.5, zorder=2) # Main out branch

# Branch to 2.2
ax.plot([0.42, 0.42], [0.55, 0.82], color="#475569", lw=2.5, zorder=2)
draw_arrow(0.42, 0.82, 0.455, 0.82)
draw_label(0.435, 0.85, "468 Facial\nLandmarks", fc="#eff6ff", ec="#cbd5e1")

# Branch to 2.3
draw_arrow(0.42, 0.64, 0.455, 0.64)

# Branch to 2.4
ax.plot([0.42, 0.42], [0.55, 0.46], color="#475569", lw=2.5, zorder=2)
draw_arrow(0.42, 0.46, 0.455, 0.46)

# Branch to 2.5
ax.plot([0.42, 0.42], [0.55, 0.28], color="#475569", lw=2.5, zorder=2)
draw_arrow(0.42, 0.28, 0.455, 0.28)
draw_label(0.435, 0.33, "Cropped\nFace Box", fc="#eff6ff", ec="#cbd5e1")

# Convergence from 2.2, 2.3, 2.4 to 2.6
ax.plot([0.625, 0.69], [0.82, 0.82], color="#475569", lw=2.5, zorder=2)
draw_label(0.655, 0.84, "EAR Float")

ax.plot([0.625, 0.69], [0.64, 0.64], color="#475569", lw=2.5, zorder=2)
draw_label(0.655, 0.66, "Yaw/Pitch")

ax.plot([0.625, 0.69], [0.46, 0.46], color="#475569", lw=2.5, zorder=2)
draw_label(0.655, 0.44, "CNN Prob")

# Connect vertical bus
ax.plot([0.69, 0.69], [0.82, 0.46], color="#475569", lw=2.5, zorder=2)

# Input into 2.6
draw_arrow(0.69, 0.55, 0.725, 0.55)

# 2.5 to DB
draw_arrow(0.54, 0.21, 0.54, 0.15)
draw_label(0.54, 0.18, "Write 128D\nFeature Vectors")

# DB to 2.6
ax.plot([0.63, 0.81], [0.08, 0.08], color="#ea580c", lw=2.5, zorder=2, ls="--")
ax.annotate("", xy=(0.81, 0.47), xytext=(0.81, 0.08), arrowprops=dict(arrowstyle="-|>", color="#ea580c", lw=2.5, ls="--"), zorder=2)
ax.text(0.81, 0.25, "Recognized\nStudent ID", ha="center", va="center", size=10, weight="bold", color="#c2410c", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"), zorder=6)

# 2.6 to Output - Extended X coordinates so it doesn't overlap!
draw_arrow(0.905, 0.55, 0.94, 0.55)
draw_label(0.922, 0.58, "Final Analyzed\nJSON Package")

ax.axis('off')
ax.set_xlim(0, 1.1)
ax.set_ylim(-0.02, 0.95)
plt.title("Level 2 Data Flow Diagram (DFD)\nProcess 2.0: Core AI Attention Engine", fontsize=24, weight='bold', pad=20, color="#1e293b")
plt.tight_layout()

# Save image
output_path = "DFD_Level2_Visual.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False, facecolor="white")
print(f"Level 2 DFD successfully generated at: {output_path}")
