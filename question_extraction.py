"""Ce fichier contient le module pour générer des questions
"""
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class QuestionExtractor:
    """ Cette classe contient toutes les méthodes
    requises pour extraire des questions d'un document donné
    """

    def __init__(self, num_questions):
        self.num_questions = num_questions

        # hash set pour une recherche rapide
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Téléchargement des données NLTK...")
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))

        # tagueur de reconnaissance d'entités nommées
        try:
            self.ner_tagger = spacy.load('en_core_web_md')
        except OSError:
            print("Modèle spaCy 'en_core_web_md' non trouvé.")
            print("Installez-le avec: python -m spacy download en_core_web_md")
            try:
                # Essayer le modèle plus petit en fallback
                self.ner_tagger = spacy.load('en_core_web_sm')
                print("Utilisation du modèle 'en_core_web_sm' à la place.")
            except OSError:
                print("Aucun modèle spaCy trouvé. Installez au moins 'en_core_web_sm'")
                raise

        self.vectorizer = TfidfVectorizer()
        self.questions_dict = dict()

    def get_questions_dict(self, document):
        """
        Retourne un dict de questions au format:
        question_number: {
            question: str
            answer: str
        }

        Params:
            * document : string
        Returns:
            * dict
        """
        try:
            # trouver les mots-clés candidats
            self.candidate_keywords = self.get_candidate_entities(document)

            if not self.candidate_keywords:
                print("Aucune entité trouvée dans le document")
                return {}

            # définir les scores des mots avant de classer les mots-clés candidats
            self.set_tfidf_scores(document)

            # classer les mots-clés en utilisant les scores tf idf calculés
            self.rank_keywords()

            # former les questions
            self.form_questions()

            return self.questions_dict
        except Exception as e:
            print(f"Erreur dans get_questions_dict: {e}")
            return {}

    def get_filtered_sentences(self, document):
        """ Retourne une liste de phrases - chacune
        ayant été nettoyée des mots vides.
        Params:
                * document: un paragraphe de phrases
        Returns:
                * list<str> : liste de chaînes
        """
        sentences = sent_tokenize(document)  # diviser les documents en phrases
        return [self.filter_sentence(sentence) for sentence in sentences]

    def filter_sentence(self, sentence):
        """Retourne la phrase sans les mots vides
        Params:
                * sentence: Une chaîne
        Returns:
                * string
        """
        words = word_tokenize(sentence)
        return ' '.join(w for w in words if w.lower() not in self.stop_words)

    def get_candidate_entities(self, document):
        """ Retourne une liste d'entités selon
        le tagueur ner de spacy. Ces entités sont candidates
        pour les questions

        Params:
                * document : string
        Returns:
                * list<str>
        """
        try:
            entities = self.ner_tagger(document)
            entity_list = []

            for ent in entities.ents:
                # Filtrer les entités trop courtes ou peu significatives
                if len(ent.text.strip()) > 1 and ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'PERCENT']:
                    entity_list.append(ent.text.strip())

            return list(set(entity_list))  # supprimer les doublons
        except Exception as e:
            print(f"Erreur lors de l'extraction des entités: {e}")
            return []

    def set_tfidf_scores(self, document):
        """ Définit les scores tf-idf pour chaque mot"""
        try:
            self.unfiltered_sentences = sent_tokenize(document)
            self.filtered_sentences = self.get_filtered_sentences(document)

            if not self.filtered_sentences:
                print("Aucune phrase filtrée trouvée")
                return

            self.word_score = dict()  # (word, score)

            # (word, sentence where word score is max)
            self.sentence_for_max_word_score = dict()

            tf_idf_vector = self.vectorizer.fit_transform(self.filtered_sentences)

            # Gestion des différentes versions de sklearn
            try:
                feature_names = self.vectorizer.get_feature_names_out()
            except AttributeError:
                feature_names = self.vectorizer.get_feature_names()

            tf_idf_matrix = tf_idf_vector.todense().tolist()

            num_sentences = len(self.unfiltered_sentences)
            num_features = len(feature_names)

            for i in range(num_features):
                word = feature_names[i]
                self.sentence_for_max_word_score[word] = ""
                tot = 0.0
                cur_max = 0.0

                for j in range(num_sentences):
                    if j < len(tf_idf_matrix):
                        tot += tf_idf_matrix[j][i]

                        if tf_idf_matrix[j][i] > cur_max:
                            cur_max = tf_idf_matrix[j][i]
                            self.sentence_for_max_word_score[word] = self.unfiltered_sentences[j]

                # score moyen pour chaque mot
                self.word_score[word] = tot / num_sentences if num_sentences > 0 else 0
        except Exception as e:
            print(f"Erreur lors du calcul des scores TF-IDF: {e}")

    def get_keyword_score(self, keyword):
        """ Retourne le score pour un mot-clé
        Params:
            * keyword : string de possibles plusieurs mots
        Returns:
            * float : score
        """
        score = 0.0
        words = word_tokenize(keyword.lower())
        for word in words:
            if word in self.word_score:
                score += self.word_score[word]
        return score

    def get_corresponding_sentence_for_keyword(self, keyword):
        """ Trouve et retourne une phrase contenant
        les mots-clés
        """
        words = word_tokenize(keyword.lower())
        for word in words:
            if word not in self.sentence_for_max_word_score:
                continue

            sentence = self.sentence_for_max_word_score[word]

            all_present = True
            for w in words:
                if w.lower() not in sentence.lower():
                    all_present = False
                    break

            if all_present:
                return sentence
        return ""

    def rank_keywords(self):
        """Classer les mots-clés selon leur score"""
        self.candidate_triples = []  # (score, keyword, corresponding sentence)

        for candidate_keyword in self.candidate_keywords:
            sentence = self.get_corresponding_sentence_for_keyword(candidate_keyword)
            if sentence:  # Seulement ajouter si une phrase correspondante est trouvée
                self.candidate_triples.append([
                    self.get_keyword_score(candidate_keyword),
                    candidate_keyword,
                    sentence
                ])

        self.candidate_triples.sort(reverse=True)

    def form_questions(self):
        """ Forme les questions et remplit
        le dict des questions
        """
        used_sentences = list()
        idx = 0
        cntr = 1
        num_candidates = len(self.candidate_triples)

        while cntr <= self.num_questions and idx < num_candidates:
            candidate_triple = self.candidate_triples[idx]

            if candidate_triple[2] not in used_sentences and candidate_triple[2].strip():
                used_sentences.append(candidate_triple[2])

                # Créer la question en remplaçant le mot-clé par des tirets
                question_text = candidate_triple[2].replace(
                    candidate_triple[1],
                    '_' * len(candidate_triple[1]))

                self.questions_dict[cntr] = {
                    "question": question_text,
                    "answer": candidate_triple[1]
                }

                cntr += 1
            idx += 1