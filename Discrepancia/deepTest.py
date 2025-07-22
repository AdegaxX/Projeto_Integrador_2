from deepface import DeepFace

with open("deepface_output.txt", "w", encoding="utf-8") as f:
    f.write("🔍 Comparando rostos com DeepFace...\n")
    try:
        result = DeepFace.verify(img1_path="temp1.jpg", img2_path="temp2.jpg", enforce_detection=False)
        distance = result["distance"]
        f.write(f"📏 Distância facial: {distance:.4f}\n")
        if distance > 0.6:
            f.write("⚠️ A pessoa está diferente. Pode ser necessário refazer o RG.\n")
        else:
            f.write("✅ As imagens são semelhantes. Não parece necessário refazer o RG.\n")
    except Exception as e:
        f.write(f"❌ Erro ao comparar rostos: {e}\n")
