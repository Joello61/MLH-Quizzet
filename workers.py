"""
Module workers.py mis à jour avec meilleure gestion d'erreurs
"""

try:
    from PyPDF2 import PdfReader  # Version récente
except ImportError:
    try:
        from PyPDF2 import PdfFileReader as PdfReader  # Version ancienne comme la vôtre
    except ImportError:
        print("Erreur : PyPDF2 non trouvé. Installez avec: pip install PyPDF2")
        raise

from question_generation_main import QuestionGeneration


def pdf2text(file_path: str, file_exten: str) -> str:
    """Convertit un fichier donné en contenu texte"""

    _content = ''

    try:
        # Identifier le type de fichier et obtenir son contenu
        if file_exten.lower() == 'pdf':
            with open(file_path, 'rb') as pdf_file:
                try:
                    # Essayer d'abord avec votre version actuelle
                    from PyPDF2 import PdfFileReader
                    _pdf_reader = PdfFileReader(pdf_file)
                    for p in range(_pdf_reader.numPages):
                        _content += _pdf_reader.getPage(p).extractText()
                except ImportError:
                    # Si la version récente est installée
                    pdf_reader = PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        _content += page.extract_text()
                except Exception as e:
                    print(f"Erreur lors de la lecture du PDF : {e}")
                    return None

            print('PDF operation done!')

        elif file_exten.lower() == 'txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    _content = txt_file.read()
            except UnicodeDecodeError:
                # Essayer avec un autre encodage si UTF-8 échoue
                try:
                    with open(file_path, 'r', encoding='latin-1') as txt_file:
                        _content = txt_file.read()
                except Exception as e:
                    print(f"Erreur lors de la lecture du fichier texte : {e}")
                    return None

            print('TXT operation done!')

        else:
            print(f"Type de fichier non supporté : {file_exten}")
            return None

    except FileNotFoundError:
        print(f"Fichier non trouvé : {file_path}")
        return None
    except Exception as e:
        print(f"Erreur inattendue lors du traitement du fichier : {e}")
        return None

    return _content if _content.strip() else None


def txt2questions(doc: str, n=5, o=4) -> dict:
    """Obtient toutes les questions et options"""

    if not doc or not doc.strip():
        print("Erreur : Document vide ou invalide")
        return {}

    try:
        qGen = QuestionGeneration(n, o)
        q = qGen.generate_questions_dict(doc)

        # Reformater les options pour l'affichage
        for i in range(len(q)):
            if i + 1 in q and 'options' in q[i + 1]:
                # Vérifier si les options sont dans le bon format
                if isinstance(q[i + 1]['options'], dict):
                    # Convertir en liste si c'est un dictionnaire
                    temp = []
                    for j in range(1, len(q[i + 1]['options']) + 1):
                        if j in q[i + 1]['options']:
                            temp.append(q[i + 1]['options'][j])
                    q[i + 1]['options'] = temp

        return q

    except Exception as e:
        print(f"Erreur lors de la génération des questions : {e}")
        return {}