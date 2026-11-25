"""Navegação interativa para executar e consultar os experimentos do projeto."""

from pathlib import Path

from experiments import evaluate_multiscale_context as e01
from experiments import manual_execution as ops
from experiments.reporting import REPORT_FILES, render_summary, saved_model_paths
from src.feature_extractor import DEFAULT_FEATURE_GROUPS, FEATURE_GROUPS, canonical_feature_groups


DATASET_TARGETS = ("train.npz", "validation.npz")
EVALUATION_TARGETS = ("configuration.json", "global_metrics.json", "per_image_metrics.csv",
                      "threshold_metrics.csv", "model_comparison.csv", "false_positives_by_class.csv",
                      "i01_superpixel_comparison.csv", "training_statistics.json", "predictions.npz", "summary.md")


class TerminalMenu:
    def __init__(self, ask=input):
        self.ask = ask

    def choose(self, title: str, options: list, *, exit_label: str = "Voltar"):
        while True:
            print(f"\n{title}")
            for number, (label, _) in enumerate(options, 1):
                print(f"{number}. {label}")
            print(f"0. {exit_label}")
            answer = self.ask("Opção: ").strip()
            if answer == "0":
                return None
            for number, (_, value) in enumerate(options, 1):
                if answer == str(number):
                    return value
            print("Opção inválida.")

    def run_path(self, namespace: str) -> Path:
        name = self.ask("Nome da execução [execucao-1]: ").strip() or "execucao-1"
        return ops.safe_run_path(namespace, name)

    def execute(self, title, output, targets, operation, *, summarize=True) -> bool:
        if not ops.allow_write(output, targets, self.ask):
            print("Operação cancelada. Artefatos preservados.")
            return False
        ops.process(title, output, operation, summarize=summarize)
        return True

    def source(self, title, paths, required):
        available = list(dict.fromkeys(path for path in paths if all((path / name).is_file() for name in required)))
        if not available:
            raise FileNotFoundError(f"Nenhuma origem disponível para {title}: {', '.join(required)}.")
        return self.choose(title, [(path.as_posix(), path) for path in available])

    def inspect(self) -> None:
        subset = self.choose("Subconjunto", [("Treino", "train"), ("Validação", "validation")])
        if subset is not None:
            answer = self.ask("Quantidade de imagens [5]: ").strip() or "5"
            if not answer.isdecimal() or not 1 <= int(answer) <= len(ops.manifest()[subset]):
                raise ValueError("Informe uma quantidade positiva dentro do subconjunto selecionado.")
            ops.inspect(subset, int(answer))

    def build(self) -> None:
        groups = self.choose_feature_groups()
        if groups is None:
            return
        output = self.run_path("baseline")
        self.execute(f"Construção · {', '.join(groups)}", output,
                     (*DATASET_TARGETS, "split.json", "dataset_statistics.json", "summary.md"),
                     lambda: ops.build_baseline_copy(output, groups))

    def choose_feature_groups(self) -> tuple[str, ...] | None:
        multiscale_lab = (*DEFAULT_FEATURE_GROUPS, "multiscale_lab")
        presets = [("B00", DEFAULT_FEATURE_GROUPS), ("E01.1-A", multiscale_lab),
                   ("E01.1-B", (*multiscale_lab, "multiscale_robust"))]
        options = [(f"{name} ({sum(len(FEATURE_GROUPS[group]) for group in groups)} features)", groups)
                   for name, groups in presets]
        selected = self.choose("Características", [*options, ("Personalizada", "custom")])
        if selected != "custom":
            return selected
        available = {}
        for index, (group, names) in enumerate(FEATURE_GROUPS.items(), 1):
            available[str(index)] = group
            print(f"{index}. {group} ({len(names)} features)")
        while True:
            answer = self.ask("Selecione grupos por número, separados por espaço (0 para voltar): ").split()
            if answer == ["0"]:
                return None
            if answer and all(index in available for index in answer):
                return canonical_feature_groups(available[index] for index in answer)
            print("Seleção inválida.")

    def train(self) -> None:
        output = self.run_path("baseline")
        saved = (path.parent for path in sorted(ops.BASELINE.parent.rglob("train.npz")))
        candidates = (output, ops.BASELINE, *saved)
        source = self.source("NPZ para treino", candidates, DATASET_TARGETS)
        if source is not None:
            targets = ("random_forest.joblib", "validation_metrics.json", "summary.md")
            if not (output / "split.json").is_file():
                targets = (*targets, "split.json")
            self.execute(f"Treino · origem {source}", output,
                         targets,
                         lambda: ops.train_baseline_copy(output, source))

    def evaluate(self, *, single: bool = False) -> None:
        root = self.run_path("baseline")
        image_id = None
        output = root
        if single:
            image_id = self.ask("ID de validação (vazio para voltar): ").strip()
            if not image_id:
                return
            if image_id not in ops.manifest()["validation"]:
                raise ValueError("O ID informado não pertence à validação.")
            output = root / "images" / image_id
            if output.resolve() != root.resolve() / "images" / image_id:
                raise ValueError("Destino da imagem redirecionado.")
        models = saved_model_paths(ops.BASELINE, ops.BASELINE.parent)
        if not models:
            raise FileNotFoundError("Nenhum modelo salvo disponível.")
        model = self.choose("Modelo da avaliação", [(path.as_posix(), path) for path in models])
        if model is None:
            return
        decisions = [("Decisão padrão do estimador", "default"), ("Limiar personalizado", "threshold")]
        if not single:
            decisions.append(("Limiar ótimo por IoU nesta validação", "optimal"))
        decision = self.choose("Decisão", decisions)
        if decision is None:
            return
        threshold = None
        if decision == "threshold":
            answer = self.ask("Limiar de 0 a 1 (vazio para voltar): ").strip().replace(",", ".")
            if not answer:
                return
            threshold = float(answer)
            if not 0 <= threshold <= 1:
                raise ValueError("O limiar deve estar entre 0 e 1.")
        if decision == "optimal":
            print("O limiar será escolhido pelas anotações desta validação. A IoU é obtida nesse mesmo conjunto.")
        targets = (("global_metrics.json", "prediction.png", "summary.md") if single else
                   ("pixel_validation_metrics.json", "pixel_validation_per_image.json", "pixel_examples", "summary.md"))
        self.execute(f"Avaliação {'individual' if single else 'da validação'} · modelo {model}", output,
                     targets, lambda: ops.evaluate_saved_model(output, model, image_id,
                                                              threshold=threshold, optimize=decision == "optimal"))

    def diagnostics(self) -> None:
        while selected := self.choose("Diagnósticos", [(f"{item.code} — {item.title}", item) for item in ops.EXPERIMENTS]):
            action = self.choose(f"{selected.code} — {selected.title}", [("Executar", "run"), ("Consultar", "read")])
            if action == "read":
                self.consult(ops.diagnostic_paths(selected.folder))
            elif action == "run":
                self.run_diagnostic(selected)

    def run_diagnostic(self, selected) -> None:
        ops.require_files(ops.BASELINE, "split.json", *selected.baseline_files)
        ops.manifest()
        dependencies = []
        for folder, required in selected.dependencies:
            choice = self.source(f"Artefatos de {folder}", ops.diagnostic_paths(folder), required)
            if choice is None:
                return
            dependencies.append(choice)
        output = self.run_path(selected.folder)
        reuse_options = {}
        if selected.code == "D04" and (output / "validation_features.npz").is_file():
            reuse_options["reuse_features"] = self.ask("Reutilizar a matriz de features deste diretório? [S/n]: ").strip().lower() != "n"
        variants = {"D01": (ops.DATASET, ops.BASELINE), "D02": (ops.DATASET, ops.BASELINE, *dependencies),
                    "D03": (*dependencies, ops.BASELINE, ops.DATASET), "D04": (ops.DATASET, ops.BASELINE, *dependencies)}
        targets = tuple(path.name for path in output.iterdir()) if output.is_dir() else ()
        self.execute(f"{selected.code} · entradas {', '.join(map(str, variants[selected.code]))}", output, targets,
                     lambda: selected.run(*variants[selected.code], output, **reuse_options))

    def optimization(self) -> None:
        experiment = self.choose("Otimização", [("E01.1 — Contexto espacial multiescala", "E01_1_multiscale_context")])
        if experiment is None:
            return
        options = [("Preparar datasets", "prepare"), ("Treinar modelos", "train"), ("Executar validação", "validate"),
                   ("Gerar análises e visualizações", "render"), ("Executar pipeline completo", "all"),
                   ("Consultar resultados existentes", "read")]
        while action := self.choose("E01.1", options):
            if action == "read":
                self.consult([ops.OPTIMIZATION, *(path for path in ops.report_paths()
                                                 if experiment in path.parts)])
            else:
                self.run_optimization(experiment, action)

    def run_optimization(self, experiment: str, action: str) -> None:
        root = self.run_path(experiment)
        if action == "prepare":
            self.prepare_optimization(root)
            return
        variants = self.choose("Variantes", [("E01.1-A", ("E01.1-A",)), ("E01.1-B", ("E01.1-B",)),
                                           ("Todas", ("E01.1-A", "E01.1-B"))])
        if variants is None:
            return
        output = root / "evaluations" / ("all" if len(variants) == 2 else variants[0])
        if output.resolve() != root.resolve() / "evaluations" / output.name:
            raise ValueError("Destino da avaliação redirecionado.")
        title = f"E01.1 · {', '.join(variants)}"
        if action == "render":
            self.render_optimization(title, output)
            return

        if action in ("validate", "all"):
            ops.require_files(ops.DIAGNOSTICS / ops.D02, "region_statistics.csv")
            ops.require_files(ops.DIAGNOSTICS / ops.D03, "global_statistics.json")
            ops.require_files(ops.DIAGNOSTICS / ops.D04, "feature_statistics_by_outcome.csv")

        prepared_candidates = (root / "datasets", ops.OPTIMIZATION / "datasets")
        required = (*DATASET_TARGETS, "preparation_statistics.json")
        if action == "all" and not any(all((path / item).is_file() for item in required) for path in prepared_candidates):
            if not self.prepare_optimization(root):
                return
        prepared = self.source("Datasets experimentais", prepared_candidates, required)
        if prepared is None:
            return
        ops.validate_datasets(prepared, experimental=True)
        model_files = tuple(f"{name}.joblib" for name in variants)
        if action == "train":
            self.train_optimization(title, root, prepared, variants)
            return
        model_candidates = (root / "models", ops.OPTIMIZATION / "models")
        if action == "all" and not any(all((path / item).is_file() for item in model_files) for path in model_candidates):
            missing = tuple(name for name in variants if not (root / "models" / f"{name}.joblib").is_file())
            if not self.train_optimization(title, root, prepared, missing):
                return
        models = self.source("Modelos experimentais salvos", model_candidates, model_files)
        if models is None:
            return
        if self.execute(f"{title} · validação com {models}", output, EVALUATION_TARGETS,
                        lambda: e01.run_evaluation(ops.DATASET, ops.BASELINE, ops.DIAGNOSTICS, output,
                                                   variants=variants, prepared=prepared, models_root=models,
                                                   reuse_models=True, render=False)) and action == "all":
            self.render_optimization(title, output)

    def prepare_optimization(self, root: Path) -> bool:
        return self.execute("E01.1 · preparação compartilhada", root / "datasets",
                            (*DATASET_TARGETS, "preparation_statistics.json", "summary.md"),
                            lambda: e01.prepare_datasets(ops.DATASET, ops.BASELINE, root))

    def train_optimization(self, title, root, prepared, variants) -> bool:
        targets = tuple(f"models/{name}.joblib" for name in variants)
        return self.execute(f"{title} · treinamento", root, (*targets, "training_statistics.json", "summary.md"),
                            lambda: e01.train_prepared(ops.BASELINE, prepared, root, variants))

    def render_optimization(self, title, output) -> bool:
        ops.require_files(output, "configuration.json", "global_metrics.json", "predictions.npz", "per_image_metrics.csv")
        return self.execute(f"{title} · visualização de predições salvas", output, ("charts", "examples"),
                            lambda: e01.render_saved_evaluation(ops.DATASET, output), summarize=False)

    def consult(self, paths) -> None:
        roots = [path for path in dict.fromkeys(paths) if any((path / name).is_file() for name in REPORT_FILES)]
        if not roots:
            print("Nenhum relatório disponível.")
            return
        chosen = self.choose("Consultar resultados", [(path.as_posix(), path) for path in roots])
        if chosen is not None:
            print(render_summary(chosen))

    def compare(self) -> None:
        paths = ops.report_paths()
        if not paths:
            print("Nenhum relatório disponível.")
            return
        first = self.choose("Primeiro resultado", [(path.as_posix(), path) for path in paths])
        if first is not None:
            second = self.choose("Segundo resultado", [(path.as_posix(), path) for path in paths if path != first])
            if second is not None:
                ops.compare_reports([first, second])

    def run(self) -> None:
        options = [("Inspecionar dataset", self.inspect), ("Construir datasets de treino e validação", self.build),
                   ("Treinar modelo", self.train), ("Executar validação", self.evaluate),
                   ("Avaliar imagem individual", lambda: self.evaluate(single=True)),
                   ("Experimentos diagnósticos", self.diagnostics), ("Experimentos de otimização", self.optimization),
                   ("Comparar resultados existentes", self.compare),
                   ("Consultar resultados existentes", lambda: self.consult(ops.report_paths()))]
        while True:
            try:
                action = self.choose("PI-Seg-Fissuras", options, exit_label="Sair")
                if action is None:
                    return
                action()
            except (EOFError, KeyboardInterrupt):
                print("\nExecução encerrada.")
                return
            except (ValueError, KeyError, OSError, RuntimeError, SystemExit) as error:
                print(f"Erro: {error}")
