import subprocess
import importlib

def install_check_version(packages):

    # Channel list for micromamba
    channels = "-c conda-forge -c bioconda -c pytorch -c nvidia -c defaults"

    for pkg in packages:
        installed = False
        print(f"\nTrying to install: {pkg}")

        # Step 1: Try micromamba
        micromamba_cmd = f"micromamba install {channels} {pkg} -y"
        result = subprocess.run(
            micromamba_cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            print(f"Installed via micromamba: {pkg}")
            installed = True
        else:
            print(f"Micromamba failed: {pkg}")

        # Step 2: Try pip
        if not installed:
            pip_cmd = f"pip install {pkg}"
            result = subprocess.run(
                pip_cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if result.returncode == 0:
                print(f"Installed via pip: {pkg}")
                installed = True
            else:
                print(f"Pip failed: {pkg}")

        # Step 3: Try brew (macOS only)
        if not installed:
            brew_cmd = f"brew install {pkg}"
            result = subprocess.run(
                brew_cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            if result.returncode == 0:
                print(f"Installed via brew: {pkg}")
                installed = True
            else:
                print(f"Brew failed: {pkg}")

        # Try importing the module (normalize some names for import)
        module_name = pkg.replace('-', '_').replace('.', '_')

        # Version fetched
        version_flag = False

        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"{pkg}: {version}")
            version_flag = True
        except ImportError:
            pass

        # Try using importlib.metadata
        if not version_flag:
            try:
                version = importlib.metadata.version(pkg)
                print(f"{pkg}: {version}")
                version_flag = True
            except Exception:
                pass

        # Try using importlib.metadata
        if not version_flag: 
            try:
                version = importlib.metadata.version(module_name)
                print(f"{pkg}: {version}")
                version_flag = True
            except Exception:
                pass 

        # Try pip show
        if not version_flag:
            try:
                result = subprocess.run(['pip', 'show', pkg], capture_output=True, text=True)
                for line in result.stdout.splitlines():
                    if line.startswith("Version:"):
                        print(f"{pkg}: {line.split(': ')[1]} (via pip)")
                        version_flag = True
            except Exception:
                pass

        # Try micromamba (or conda)
        if not version_flag:
            try:
                result = subprocess.run(['micromamba', 'list', pkg], capture_output=True, text=True)
                if pkg in result.stdout:
                    print(f"{pkg}: found via micromamba/conda")
            except Exception:
                pass

        # Try brew
        if not version_flag:
            try:
                result = subprocess.run(['brew', 'list', '--versions', pkg], capture_output=True, text=True)
                if result.stdout:
                    print(f"{pkg}: {result.stdout.strip()} (via brew)")
                    rversion_flag = True
            except Exception:
                pass

if __name__ == "__main__":

    packages_2 = [
        "huggingface_hub",
        "byted-huggingface-hub"
        "corex-ai-nlp-huggingface",
        "cria-llms-huggingface",
        "evo-package-huggingface",
        "flying-delta-embeddings-huggingface",
        "flying-delta-llms-huggingface",
        "flytekitplugins-huggingface",
        "forte-huggingface",
        "gigachain-huggingface",
        "gradio-huggingfacehub-search",
        "graphbook-huggingface",
        "huggingface",
        "huggingface-datasets-cocoapi-tools",
        "huggingface-download-cli",
        "huggingface-guess",
        "huggingface-hub",
        "huggingface-hub-storj-patch",
        "huggingfaceinference",
        "huggingface-mcp-server",
        "huggingface-nas",
        "huggingface-rdf",
        "huggingface-sb3",
        "huggingface-text-data-analyzer",
        "huggingface-togetherai",
        "huggingface-tool",
        "huggingface-urls",
        "ipfs-huggingface-scraper-py",
        "kilroy-module-huggingface",
        "kozmoserver-huggingface",
        "langchain-huggingface",
        "langevals-huggingface",
        "lavague-contexts-huggingface",
        "lavague-llms-huggingface",
        "llama-index-embeddings-huggingface",
        "llama-index-embeddings-huggingface-api",
        "llama-index-embeddings-huggingface-optimum",
        "llama-index-embeddings-huggingface-optimum-intel",
        "llama-index-llms-huggingface",
        "llama-index-llms-huggingface-api",
        "llama-index-multi-modal-llms-huggingface",
        "llama-index-readers-huggingface-fs",
        "llama-index-utils-huggingface",
        "llm-huggingface",
        "mlserver-huggingface",
        "mlserver-huggingface-striveworks",
        "pandasai-huggingface",
        "petinterfacehuggingface",
        "pyspark-huggingface",
        "roboflow2huggingface",
        "sagemaker-huggingface-inference-toolkit",
        "shiertier-huggingface",
        "simpleraghuggingface",
        "sinapsis-huggingface",
        "sinapsis-huggingface-diffusers",
        "sinapsis-huggingface-embeddings",
        "sinapsis-huggingface-grounding-dino",
        "sinapsis-huggingface-transformers",
        "spacy-huggingface-hub",
        "spacy-huggingface-pipelines",
        "streamlit-huggingface",
        "trulens-providers-huggingface",
        "vdk-huggingface",
        "vertex-ai-huggingface-inference-toolkit",
        "napari-bioimageio",
        "bioimageio-core",
        "bioimageio-engine",
        "bioimageio-spec",
        "bioimageio-model",
        "bioimageio-utils",
        "byted-huggingface-hub",
        "corex-ai-nlp-huggingface",
        "cria-llms-huggingface",
        "evo-package-huggingface",
        "flying-delta-embeddings-huggingface",
        "flying-delta-llms-huggingface",
        "flytekitplugins-huggingface",
        "forte-huggingface",
        "gigachain-huggingface",
        "gradio-huggingfacehub-search",
        "graphbook-huggingface",
        "huggingface",
        "huggingface-datasets-cocoapi-tools",
        "huggingface-download-cli",
        "huggingface-guess",
        "huggingface-hub",
        "huggingface-hub-storj-patch",
        "huggingfaceinference",
        "huggingface-mcp-server",
        "huggingface-nas",
        "huggingface-rdf",
        "huggingface-sb3",
        "huggingface-text-data-analyzer",
        "huggingface-togetherai",
        "huggingface-tool",
        "huggingface-urls",
        "ipfs-huggingface-scraper-py",
        "kilroy-module-huggingface",
        "kozmoserver-huggingface",
        "langchain-huggingface",
        "langevals-huggingface",
        "lavague-contexts-huggingface",
        "lavague-llms-huggingface",
        "llama-index-embeddings-huggingface",
        "llama-index-embeddings-huggingface-api",
        "llama-index-embeddings-huggingface-optimum",
        "llama-index-embeddings-huggingface-optimum-intel",
        "llama-index-llms-huggingface",
        "llama-index-llms-huggingface-api",
        "llama-index-multi-modal-llms-huggingface",
        "llama-index-readers-huggingface-fs",
        "llama-index-utils-huggingface",
        "llm-huggingface",
        "mlserver-huggingface",
        "mlserver-huggingface-striveworks",
        "pandasai-huggingface",
        "petinterfacehuggingface",
        "pyspark-huggingface",
        "roboflow2huggingface",
        "sagemaker-huggingface-inference-toolkit",
        "shiertier-huggingface",
        "simpleraghuggingface",
        "sinapsis-huggingface",
        "sinapsis-huggingface-diffusers",
        "sinapsis-huggingface-embeddings",
        "sinapsis-huggingface-grounding-dino",
        "sinapsis-huggingface-transformers",
        "spacy-huggingface-hub",
        "spacy-huggingface-pipelines",
        "streamlit-huggingface",
        "trulens-providers-huggingface",
        "vdk-huggingface",
        "vertex-ai-huggingface-inference-toolkit",
        "transformers",
        "datasets",
        "accelerate",
        "diffusers",
        "peft",
        "evaluate",
        "tokenizers",
        "optimum",
        "bert-tensorflow",
        "blitz-bayesian-pytorch",
        "botorch",
        "cellfinder-napari",
        "cellpose-napari",
        "cuequivariance-torch",
        "curvlinops-for-pytorch",
        "devbio-napari",
        "efficientnet-pytorch",
        "fast-pytorch-kmeans",
        "fft-conv-pytorch",
        "flowtorch",
        "galore-torch",
        "gradio-huggingfacehub-search",
        "jax2torch",
        "kilroy-module-huggingface",
        "kozmoserver-huggingface",
        "laplace-torch",
        "libmetatensor-torch",
        "libopenvino-pytorch-frontend",
        "libopenvino-tensorflow-frontend",
        "libopenvino-tensorflow-lite-frontend",
        "libtensorflow",
        "libtensorflow_cc",
        "llama-index-embeddings-huggingface",
        "llama-index-embeddings-huggingface-api",
        "llama-index-embeddings-huggingface-optimum",
        "llama-index-embeddings-huggingface-optimum-intel",
        "llama-index-llms-huggingface",
        "llama-index-llms-huggingface-api",
        "llama-index-multi-modal-llms-huggingface",
        "llama-index-readers-huggingface-fs",
        "llama-index-utils-huggingface",
        "llm-huggingface",
        "lovely-tensors",
        "mlserver-huggingface",
        "mlserver-huggingface-striveworks",
        "napari",
        "napari-3d-counter",
        "napari-3d-ortho-viewer",
        "napari-accelerated-pixel-and-object-classification",
        "napari-aicsimageio",
        "napari-aideveloper",
        "napari-allencell-annotator",
        "napari-allencell-segmenter",
        "napari-annotate",
        "napari-annotation-project",
        "napari-annotator",
        "napari-annotatorj",
        "napari-base",
        "napari-bbox",
        "napari-bioformats",
        "napari-bioimageio",
        "napari-blob-detection",
        "napari-bud-cell-segmenter",
        "napari-cellseg3d",
        "napari-czann-segment",
        "napari-merge-stardist-masks",
        "napari-sam",
        "napari-sam4is",
        "napari-segment",
        "napari-segment-anything",
        "napari-segment-blobs-and-things-with-membranes",
        "napari-serialcellpose",
        "napari-vesicles-segmentation",
        "neptune-tensorflow-keras",
        "pandasai-huggingface",
        "petinterfacehuggingface",
        "pi-vae-pytorch",
        "pytorch",
        "pytorch-3dunet",
        "pytorch-cpu",
        "pytorch-ignite",
        "pytorch-lightning",
        "pytorch-metric-learning",
        "pytorch-ranger",
        "pytorch-tabnet",
        "pytorch-tabular",
        "pytorch_geometric",
        "pytorch_geometric-graphgym",
        "pytorch_geometric-modelhub",
        "pytorch_revgrad",
        "pytorch_scatter",
        "pytorch_sparse",
        "pytorch_tabular",
        "pyspark-huggingface",
        "pytest-pytorch",
        "python-metatensor-learn",
        "pycmtensor",
        "pytensor",
        "pytensor-base",
        "pyxtensor",
        "r-nntensor",
        "r-safetensors",
        "r-tensor",
        "r-tensorflow",
        "roboflow2huggingface",
        "sagemaker-huggingface-inference-toolkit",
        "sagemaker-tensorflow-container",
        "safetensors",
        "segmentation-models-pytorch",
        "shiertier-huggingface",
        "silence-tensorflow",
        "simpleraghuggingface",
        "sinapsis-huggingface",
        "sinapsis-huggingface-diffusers",
        "sinapsis-huggingface-embeddings",
        "sinapsis-huggingface-grounding-dino",
        "sinapsis-huggingface-transformers",
        "spacy-huggingface-hub",
        "spacy-huggingface-pipelines",
        "stardist-napari",
        "streamlit-huggingface",
        "sympytensor",
        "tango-pytorch_lightning",
        "tango-torch",
        "tensorboard",
        "tensorboard-plugin-wit",
        "tensorboardx",
        "tensorizer",
        "tensorly",
        "tensorly-torch",
        "tensorpac",
        "tensorpack",
        "tensordict",
        "tensorflow",
        "tensorflow-base",
        "tensorflow-cpu",
        "tensorflow-datasets",
        "tensorflow-estimator",
        "tensorflow-hub",
        "tensorflow-lattice",
        "tensorflow-privacy",
        "tensorflow-probability",
        "tensorflowonspark",
        "torch-geometric",
        "torch-hd",
        "torch-nl",
        "torch-optimizer",
        "torch-runstats",
        "torch-scatter",
        "torch-simplify",
        "torch-tb-profiler",
        "torch_em",
        "torchbiggraph",
        "torchdata",
        "torchinfo",
        "torchio",
        "torchmanager",
        "torchmd-net",
        "torchmetrics",
        "torchseg",
        "torchsnapshot",
        "torchsparse",
        "torchts",
        "torchtuples",
        "torchview",
        "torchvision",
        "torchvision-cpu",
        "torchvision-extra-decoders",
        "torchvision-tests",
        "torchx",
        "vector-quantize-pytorch",
        "vertex-ai-huggingface-inference-toolkit",
        "vit-pytorch",
        "xtensor",
        "xtensor-blas",
        "xtensor-python"
    ]

    # Call function
    install_check_version(packages_2)


