"""
Module workers.py — Lecture de fichier & génération de questions avec gestion robuste des erreurs
"""

try:
    from PyPDF2 import PdfReader  # Version récente
except ImportError:
    try:
        from PyPDF2 import PdfFileReader as PdfReader  # Version ancienne
    except ImportError:
        print(" [ERREUR IMPORT] Le module PyPDF2 est introuvable.\n"
              " Veuillez l’installer avec la commande : pip install PyPDF2")
        raise

from question_generation_main import QuestionGeneration


def generer_questions(file_path: str, file_exten: str, n=5, o=4) -> dict:
    """Lit un fichier PDF ou TXT, extrait son contenu et génère des questions à choix multiples."""

    if not file_path:
        print("[ERREUR] Aucun chemin de fichier fourni.")
        return {}

    content = ''

    # Lecture du fichier
    try:
        if file_exten.lower() == 'pdf':
            try:
                with open(file_path, 'rb') as pdf_file:
                    try:
                        from PyPDF2 import PdfFileReader
                        reader = PdfFileReader(pdf_file)
                        for p in range(reader.numPages):
                            try:
                                content += reader.getPage(p).extractText()
                            except Exception as e:
                                print(f"[PDF - Page {p}] Échec d'extraction du texte : {e}")
                    except ImportError:
                        from PyPDF2 import PdfReader
                        reader = PdfReader(pdf_file)
                        for idx, page in enumerate(reader.pages):
                            try:
                                content += page.extract_text()
                            except Exception as e:
                                print(f"[PDF - Page {idx}] Échec d'extraction du texte : {e}")
            except Exception as e:
                print(f"[ERREUR OUVERTURE PDF] {e}")
                return {}

        elif file_exten.lower() == 'txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    content = txt_file.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as txt_file:
                        content = txt_file.read()
                except Exception as e:
                    print(f"[ERREUR TXT - Encodage alternatif] {e}")
                    return {}
            except Exception as e:
                print(f"[ERREUR TXT] {e}")
                return {}
        else:
            print(f"[ERREUR TYPE] Format non supporté : {file_exten}")
            return {}

    except Exception as e:
        print(f"[ERREUR FICHIER] Problème lors de la lecture du fichier : {e}")
        return {}

    # Vérification du contenu
    if not content.strip():
        print("[ERREUR CONTENU] Aucun contenu lisible trouvé dans le fichier.")
        return {}
    # Génération des questions
    try:
        qGen = QuestionGeneration(n, o)
        q = qGen.generate_questions_dict(content)

        # Reformater les options
        for key in q:
            if 'options' in q[key] and isinstance(q[key]['options'], dict):
                q[key]['options'] = [q[key]['options'][j] for j in sorted(q[key]['options']) if j in q[key]['options']]

        return q

    except Exception as e:
        print(f"[ERREUR QUESTIONS] Impossible de générer les questions : {e}")
        return {}
