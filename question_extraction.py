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
                text = ent.text.strip()

                # Filtrage amélioré des entités
                if (len(text) > 2 and                           # Plus de 2 caractères
                    ent.label_ in [
                        'PERSON',      # Personnes
                        'ORG',         # Organisations
                        'GPE',         # Pays, villes, états
                        'DATE',        # Dates
                        'MONEY',       # Montants
                        'PERCENT',     # Pourcentages
                        'EVENT',       # Événements (batailles, révolutions, etc.)
                        'PRODUCT',     # Produits, inventions
                        'WORK_OF_ART', # Œuvres d'art, livres, films
                        'LAW',         # Lois, traités
                        'NORP',        # Nationalités, groupes religieux/politiques
                        'FACILITY',    # Bâtiments, aéroports, ponts
                        'LANGUAGE'     # Langues
                    ] and
                    not text.isdigit() and                      # Éviter les nombres seuls
                    len(text.split()) <= 4 and                  # Max 4 mots
                    not text.lower() in ['le', 'la', 'les', 'de', 'du', 'des', 'un', 'une'] # Éviter articles français
                ):
                    entity_list.append(text)

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
        ''' Forms the question and populates
        the question dict with improved formatting
        '''
        used_sentences = list()
        idx = 0
        cntr = 1
        num_candidates = len(self.candidate_triples)

        while cntr <= self.num_questions and idx < num_candidates:
            candidate_triple = self.candidate_triples[idx]
            sentence = candidate_triple[2]
            keyword = candidate_triple[1]

            if sentence not in used_sentences and sentence.strip():
                used_sentences.append(sentence)

                # Amélioration : génération de question plus naturelle
                question_text = self.create_better_question(sentence, keyword)

                self.questions_dict[cntr] = {
                    "question": question_text,
                    "answer": keyword
                }
                cntr += 1
            idx += 1

    def create_better_question(self, sentence, keyword):
        '''Crée une question plus naturelle'''

        # Calculer la longueur du blanc proportionnelle au mot-clé
        blank_length = max(5, min(15, len(keyword) + 2))
        blank = '_' * blank_length

        # Si la phrase contient déjà des mots interrogatifs, garder la structure
        question_words = ['qui', 'que', 'quoi', 'où', 'quand', 'comment', 'pourquoi',
                          'who', 'what', 'where', 'when', 'how', 'why']

        sentence_lower = sentence.lower()

        # Vérifier si c'est déjà une question
        if any(word in sentence_lower for word in question_words) or sentence.endswith('?'):
            return sentence.replace(keyword, blank)

        # Sinon, créer une question à trous avec une longueur de blanc appropriée
        question = sentence.replace(keyword, blank)

        # S'assurer que la question se termine par un point d'interrogation
        if not question.endswith('?'):
            if question.endswith('.'):
                question = question[:-1] + ' ?'
            else:
                question += ' ?'

        return question