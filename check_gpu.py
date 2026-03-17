"""
Script de vérification de la configuration GPU
Exécutez ce script pour voir si votre GPU est détecté
"""

import torch
import sys

print("="*80)
print("VERIFICATION DE LA CONFIGURATION GPU")
print("="*80)

# Version PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"Python version: {sys.version.split()[0]}")

# CUDA
print(f"\nCUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"\nGPU detecte:")
    for i in range(torch.cuda.device_count()):
        print(f"   [{i}] {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"       Memoire totale: {props.total_memory / 1024**3:.2f} GB")
        print(f"       Compute capability: {props.major}.{props.minor}")
    
    # Test GPU
    print(f"\nTest GPU:")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"   Multiplication matricielle GPU reussie!")
        print(f"   Resultat shape: {z.shape}")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   Erreur lors du test GPU: {e}")
    
    # Mémoire GPU
    print(f"\nUtilisation memoire GPU:")
    print(f"   Allouee: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"   Reservee: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    print(f"\nVotre GPU est pret pour l'entrainement RL!")
    print(f"Le notebook utilisera automatiquement le GPU.")
    
else:
    print(f"   Aucun GPU detecte")
    print(f"\nPour installer PyTorch avec GPU:")
    print(f"   1. Verifiez que vous avez un GPU NVIDIA (nvidia-smi)")
    print(f"   2. Desinstallez PyTorch CPU:")
    print(f"      pip uninstall torch torchvision torchaudio")
    print(f"   3. Installez PyTorch GPU (CUDA 11.8):")
    print(f"      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print(f"\n   Voir INSTALL_GPU.md pour plus de details")
    print(f"\nLe notebook fonctionnera quand meme avec CPU (mais plus lent)")

print("\n" + "="*80)
