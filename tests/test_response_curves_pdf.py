import matplotlib.pyplot as plt
from pathlib import Path
from scripts.visualization import compile_response_curves_pdf

def test_compile_response_curves_pdf(tmp_path):
    # Create dummy response curve images
    for i in range(3):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [i, i + 1])
        fig.savefig(tmp_path / f"var{i}_response.png")
        plt.close(fig)

    out_file = tmp_path / "combined.pdf"
    compile_response_curves_pdf(tmp_path, out_file)

    assert out_file.exists()
    assert out_file.stat().st_size > 0
