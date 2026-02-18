# OTF_3DGS

Projet de Gaussian Splatting haute performance utilisant **Burn** et **CubeCL**. Intègre **Rerun** pour la visualisation 3D en temps réel.

## Structure du projet

- `crates/otf-app` : Application principale.
- `crates/otf-kernels` : Bibliothèque de kernels GPU.

## Commandes utiles

### Compiler
Pour vérifier la compilation de tous les membres du workspace :
```bash
cargo check
```

Pour compiler le projet complet en mode debug :
```bash
cargo build
```

### Exécuter
Pour lancer l'application principale :
```bash
cargo run -p otf-app
```

### Tester
Pour lancer les tests d'un package spécifique (ex: les kernels) :
```bash
cargo test -p otf-kernels
```



