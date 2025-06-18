# Lógica de comparação e geração de relatório

def avaliar_discrepancia(distancia, atributos, limiar=0.2):
    if distancia > limiar:
        if atributos:
            return "Diferença facial significativa. Remova: " + ", ".join(atributos)
        return "Diferença facial significativa. Atualize sua foto no RG."
    else:
        if atributos:
            return "Rosto similar, mas considere remover: " + ", ".join(atributos)
        return "Foto válida, sem alterações necessárias."




