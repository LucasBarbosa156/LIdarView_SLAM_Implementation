import numpy as np


class LoopClosureDetector:
    def __init__(self, threshold=0.15, min_gap=50):

        # threshold: distância máxima para considerar loop closure
        # min_gap: número mínimo de frames de separação para evitar auto-detecção

        self.descriptors = []
        self.threshold = threshold
        self.min_gap = min_gap

    def add_scan(self, points):

        # Adiciona um novo scan e busca por loops no passado.
        # Gerar o descritor para o scan atual
        sc_query = compute_scan_context(points)
        self.descriptors.append(sc_query)
        current_idx = len(self.descriptors) - 1

        # Verificar se temos histórico suficiente
        if current_idx < self.min_gap:
            return False, -1

        best_dist = float('inf')
        best_idx = -1

        # Comparar com frames passados (respeitando o gap)
        # Buscamos do frame 0 até (atual - min_gap)
        for i in range(current_idx - self.min_gap):
            sc_candidate = self.descriptors[i]

            # Testamos todos os shifts (setores) para encontrar o melhor alinhamento
            num_sectors = sc_query.shape[1]
            current_best_shift_dist = float('inf')

            for shift in range(num_sectors):
                # Rotaciona as colunas do descritor
                sc_query_shifted = np.roll(sc_query, shift, axis=1)

                dist = scan_context_distance(sc_query_shifted, sc_candidate)

                if dist < current_best_shift_dist:
                    current_best_shift_dist = dist

            # Atualiza o melhor global se este candidato for o mais parecido até agora
            if current_best_shift_dist < best_dist:
                best_dist = current_best_shift_dist
                best_idx = i

        # 4. Verificar se o melhor match passa no critério de distância
        if best_dist < self.threshold:
            print(
                f"\n[LOOP CLOSURE] Detectado entre frame {current_idx} e {best_idx}!")
            print(f"Distância: {best_dist:.4f}")
            return True, best_idx

        return False, -1


def compute_scan_context(points, num_rings=20, num_sectors=60, max_range=80.0):

    # points: np.array (N, 3) — nuvem de pontos
    # Retorna: - sc: np.array (num_rings, num_sectors) — o descritor 2D

    # Calcular a distância horizontal (radial) e o ângulo (azimutal)
    # r = sqrt(x^2 + y^2)
    xy = points[:, :2]
    r = np.linalg.norm(xy, axis=1)

    # theta = arctan2(y, x) -> resultado em [-pi, pi]
    theta = np.arctan2(points[:, 1], points[:, 0])

    # Filtrar pontos dentro do range máximo
    mask = r < max_range
    r = r[mask]
    theta = theta[mask]
    z = points[mask, 2]

    # Converter theta para [0, 2pi]
    theta[theta < 0] += 2 * np.pi

    # Mapear r e theta para os índices da matriz (bins)
    # Anel: [0, max_range] -> [0, num_rings - 1]
    ring_idx = np.floor(r / max_range * num_rings).astype(int)

    # Setor: [0, 2pi] -> [0, num_sectors - 1]
    sector_idx = np.floor(theta / (2 * np.pi) * num_sectors).astype(int)

    # Garantir que os índices não estourem o limite (caso r == max_range)
    ring_idx = np.clip(ring_idx, 0, num_rings - 1)
    sector_idx = np.clip(sector_idx, 0, num_sectors - 1)

    # Criar o descritor e preencher com a altura máxima (z)
    sc = np.zeros((num_rings, num_sectors))
    np.maximum.at(sc, (ring_idx, sector_idx), z)

    return sc


def scan_context_distance(sc1, sc2):

    # sc1, sc2: np.array (num_rings, num_sectors)
    # Retorna: float (distância entre 0 e 1, onde 0 é idêntico)
    # Calcular o produto interno (dot product) coluna a coluna
    # Multiplicação elemento a elemento e soma no eixo das linhas (rings)

    dot_products = np.sum(sc1 * sc2, axis=0)

    # Calcular as normas de cada coluna para os dois descritores
    norm1 = np.linalg.norm(sc1, axis=0)
    norm2 = np.linalg.norm(sc2, axis=0)

    # Calcular a similaridade cosseno por setor
    # Adicionar um epsilon (1e-9) para evitar divisão por zero em setores vazios
    denominator = norm1 * norm2

    # Onde o denominador é > 0, calculamos dot/den. Onde é 0, a similaridade é 0.
    similarities = np.divide(dot_products, denominator,
                             out=np.zeros_like(dot_products),
                             where=denominator > 0)

    # A similaridade média entre todos os setores
    mean_similarity = np.mean(similarities)

    # Retorna a distância (1 - similaridade)
    # 0.0 -> Scans idênticos
    # 1.0 -> Scans totalmente diferentes
    return 1.0 - mean_similarity
