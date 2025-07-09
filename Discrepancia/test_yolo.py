from attribute_detector import detect_attributes

img_path = r"C:\Adegax\Ciência de dados - ADEGAS\5º semestre\Projeto integrador II\Imagens\selfs\LeandroAdegas.jpg"
atributos = detect_attributes(img_path)
print("Atributos detectados:", atributos)
