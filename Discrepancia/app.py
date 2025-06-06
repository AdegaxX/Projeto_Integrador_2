import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from face_utils import get_embeddings, calculate_distance
from attribute_detector import detectar_atributos
from evaluator import avaliar_discrepancia

def salvar_arquivo_utf8(caminho, conteudo):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)

def selecionar_imagem(tipo):
    caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png *.jpeg")])
    if caminho:
        destino = os.path.join("images", tipo, os.path.basename(caminho))
        shutil.copy(caminho, destino)
        messagebox.showinfo("Imagem adicionada", f"{tipo.upper()} adicionada com sucesso!")

def processar_imagens():
    rg_dir = "images/rg"
    selfie_dir = "images/selfie"
    output_dir = "relatorios"
    os.makedirs(output_dir, exist_ok=True)

    for rg_img in os.listdir(rg_dir):
        nome_base = os.path.splitext(rg_img)[0]
        selfie_img = nome_base + ".jpg"
        caminho_selfie = os.path.join(selfie_dir, selfie_img)
        caminho_rg = os.path.join(rg_dir, rg_img)

        if os.path.exists(caminho_selfie):
            try:
                emb_rg = get_embeddings(caminho_rg)
                emb_selfie = get_embeddings(caminho_selfie)
                distancia = calculate_distance(emb_rg, emb_selfie)
                atributos = detectar_atributos(caminho_selfie)
                recomendacao = avaliar_discrepancia(distancia, atributos)
                relatorio = f"Usuário: {nome_base}\nDistância facial: {distancia:.4f}\nAcessórios detectados: {', '.join(atributos) if atributos else 'Nenhum'}\nRecomendação: {recomendacao}"
                salvar_arquivo_utf8(os.path.join(output_dir, f"relatorio_{nome_base}.txt"), relatorio)
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao processar {rg_img}: {str(e)}")
        else:
            messagebox.showwarning("Imagem ausente", f"Selfie para {rg_img} não encontrada.")

    messagebox.showinfo("Concluído", "Processamento finalizado.")

janela = tk.Tk()
janela.title("Detector de Discrepância RG x Selfie")
janela.geometry("400x200")

btn_rg = tk.Button(janela, text="Selecionar imagem do RG", command=lambda: selecionar_imagem("rg"))
btn_rg.pack(pady=10)

btn_selfie = tk.Button(janela, text="Selecionar selfie atual", command=lambda: selecionar_imagem("selfie"))
btn_selfie.pack(pady=10)

btn_processar = tk.Button(janela, text="Processar imagens", command=processar_imagens)
btn_processar.pack(pady=20)

janela.mainloop()
