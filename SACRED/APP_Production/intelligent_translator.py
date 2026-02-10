#!/usr/bin/env python3
"""
Advanced Conversational AI for Tigrinya speakers
Provides sophisticated medical and agricultural knowledge
Built on 142K SQL database with intelligent reasoning
Enhanced with Wikipedia knowledge search for comprehensive answers
"""
import sqlite3
import re
import subprocess
import sys
try:
    import psutil
except ImportError:
    psutil = None
import threading
import time
import json
from pathlib import Path

# Add knowledge directory to path for Wikipedia search
sys.path.insert(0, str(Path(__file__).parent.parent / "educational_archive" / "knowledge"))

try:
    from wikipedia_search import WikipediaKnowledgeSearch, format_for_context
    WIKIPEDIA_AVAILABLE = True
    wikipedia_import_error = None
except ImportError as e:
    WIKIPEDIA_AVAILABLE = False
    wikipedia_import_error = str(e)
    print(f"Warning: Wikipedia search not available - {e}")

try:
    from context_manager import get_context_manager, format_context_status, get_model_limit
    CONTEXT_MANAGER_AVAILABLE = True
except ImportError:
    CONTEXT_MANAGER_AVAILABLE = False
    print("Warning: Context manager not available")

# Enhanced medical search integration - DISABLED FOR PERFORMANCE
# These modules don't exist and cause 90+ second delays
# try:
#     from enhanced_medical_search import search_medical_wikipedia_context
#     ENHANCED_MEDICAL_SEARCH_AVAILABLE = True
# except ImportError:
ENHANCED_MEDICAL_SEARCH_AVAILABLE = False
print("[INFO] Enhanced medical search disabled for performance")

# Enhanced African/Arab search integration - DISABLED FOR PERFORMANCE
# These modules don't exist and cause delays
# try:
#     from enhanced_african_arab_search import search_african_arab_wikipedia_context
#     ENHANCED_AFRICAN_ARAB_SEARCH_AVAILABLE = True
# except ImportError:
ENHANCED_AFRICAN_ARAB_SEARCH_AVAILABLE = False
print("[INFO] Enhanced African/Arab search disabled for performance")

try:
    from path_config import TIGRINYA_DB, WIKIPEDIA_DB, KNOWLEDGE_DIR
    PATH_CONFIG_AVAILABLE = True
except ImportError:
    PATH_CONFIG_AVAILABLE = False
    print("Warning: Path config not available, using fallback paths")

class IntelligentTigrinyaTranslator:
    """Smart phrase-matching translator using massive SQL database

    CONTEXT FOR AI ASSISTANT:
    This translator has access to a massive SQL database with 142,994 Tigrinya words and 35,899 English translations.
    When users ask questions in Tigrinya, it:
    1. Analyzes each word using SQL lookups for meaning
    2. Understands the intent (medical, agricultural, greeting, general)
    3. Provides appropriate responses in both languages

    TOOLS AVAILABLE:
    - SQL database with Tigrinya-English translations
    - Domain-specific knowledge (medical, agricultural, technical)
    - Phrase pattern matching for intent recognition
    - Context-aware response generation

    USAGE EXAMPLES:
    - Medical: "ሕማም ርእሲ እንታይ የሕክም?" → Provides headache treatment advice
    - Agricultural: Questions about farming → Farming guidance
    - General: Any Tigrinya text → Translation and helpful response

    The system is designed for real-time, intelligent conversation assistance.
    """

    def __init__(self):
        # Use path_config if available, otherwise fall back to relative path
        if PATH_CONFIG_AVAILABLE:
            self.db_path = TIGRINYA_DB
        else:
            self.db_path = Path(__file__).parent.parent / "organized_data" / "databases" / "massive_tigrinya_database.db"

        # Local AI model setup
        self.use_local_ai = True
        self.local_ai_available = self._check_local_model()

        if self.local_ai_available:
            print("[AI] Local AI model available for enhanced responses")
        else:
            print("[WARN] Local AI not available - using SQL + pattern matching only")
            self.use_local_ai = False

        # Wikipedia search initialization
        self.wikipedia_search = None
        self.wikipedia_available = False

        # Progressive loading cache initialization
        self.followup_cache = {}
        self.max_cache_size = 100  # Prevent memory overflow

        try:
            if WIKIPEDIA_AVAILABLE:
                try:
                    # Use path_config if available, otherwise fall back to relative path
                    if PATH_CONFIG_AVAILABLE:
                        wiki_db_path = WIKIPEDIA_DB
                    else:
                        wiki_db_path = Path(__file__).parent.parent / "educational_archive" / "knowledge" / "massive_wikipedia.db"

                    self.wikipedia_search = WikipediaKnowledgeSearch(str(wiki_db_path))

                    self.wikipedia_available = self.wikipedia_search.conn is not None
                    if self.wikipedia_available:
                        stats = getattr(self.wikipedia_search, 'stats', {})
                        print(f"[WIKI] Wikipedia search enabled ({stats.get('quality_articles', 0)} quality articles)")
                except Exception as e:
                    print(f"[ERROR] Wikipedia search initialization failed: {e}")
                    self.wikipedia_available = False
                    self.wikipedia_search = None
            else:
                self.wikipedia_available = False
                self.wikipedia_search = None
        except Exception as outer_e:
            print(f"[ERROR] Wikipedia initialization crashed: {outer_e}")
            self.wikipedia_available = False
            self.wikipedia_search = None

        self.medical_responses = self.load_medical_responses()
        self.agricultural_responses = self.load_agricultural_responses()
        self.medical_knowledge = self.load_medical_knowledge()
        self.agricultural_knowledge = self.load_agricultural_knowledge()
        self.regional_knowledge = self.load_regional_knowledge()

        # Load structured knowledge cards from JSON files
        self.medical_knowledge_json = self._load_knowledge_json('medical_knowledge.json')
        self.agricultural_knowledge_json = self._load_knowledge_json('agricultural_knowledge.json')
        self.mental_health_knowledge_json = self._load_knowledge_json('mental_health_knowledge.json')

    def _check_local_model(self):
        """Check if local AI models are available via Ollama"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Check for any available models (DeepSeek, Qwen, etc.)
                output = result.stdout.lower()
                return ('qwen' in output or
                        'deepseek' in output or
                        'abliterated' in output or
                        'uncensored' in output)
            return False
        except (OSError, subprocess.SubprocessError):
            return False

    def _load_knowledge_json(self, filename):
        """Load structured knowledge from JSON files for Quick Reference cards"""
        try:
            # Try organized_data/config/ path first
            config_path = Path(__file__).parent.parent / "organized_data" / "config" / filename
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"[CARDS] Loaded knowledge cards from {filename}")
                    return data
            else:
                print(f"[WARN] Knowledge file not found: {config_path}")
                return {}
        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            return {}

    def extract_knowledge_cards(self, user_input, domain=None):
        """Extract Quick Reference cards based on keyword matching in user query.

        Returns structured card data for immediate display alongside AI response.
        Cards provide instant actionable info (dosages, danger signs, planting times).
        """
        cards = []
        text_lower = user_input.lower()

        # Medical keyword detection
        medical_keywords = {
            'fever': ['fever', 'temperature', 'hot', 'burning up', 'ሓሙሽሽ', 'ረስኒ'],
            'headache': ['headache', 'head pain', 'head ache', 'ሕማም ርእሲ', 'ርእሲ'],
            'diarrhea': ['diarrhea', 'diarrhoea', 'loose stool', 'watery stool', 'ውጽኢት'],
            'cough': ['cough', 'coughing', 'ሰዓል'],
            'malaria': ['malaria', 'ወባ'],
            'dehydration': ['dehydration', 'dehydrated', 'thirst', 'ጽምኢ'],
            'bleeding': ['bleeding', 'blood', 'ደም']
        }

        # Agricultural keyword detection
        agricultural_keywords = {
            'teff': ['teff', 'ጤፍ', 'injera'],
            'sorghum': ['sorghum', 'ማሽላ'],
            'barley': ['barley', 'ገብስ'],
            'wheat': ['wheat', 'ስንዴ'],
            'cattle': ['cattle', 'cow', 'cows', 'livestock', 'ከብቲ'],
            'goats': ['goat', 'goats', 'ኣጣል'],
            'drought': ['drought', 'dry season', 'no rain', 'ድርቂ'],
            'planting': ['planting', 'plant', 'grow', 'sow', 'ምዝራእ']
        }

        # Mental health keyword detection - PRIORITY CHECK (before medical)
        # Crisis keywords get highest priority
        mental_health_crisis_keywords = {
            'suicidal_thoughts': ['suicidal', 'suicide', 'want to die', 'kill myself', 'end my life',
                                  'no reason to live', 'hurt myself', 'self-harm', 'ገዛእ ነብሲ ምቕታል'],
        }

        mental_health_keywords = {
            'depression': ['depressed', 'depression', 'sad', 'hopeless', 'worthless', 'no energy',
                          'can\'t sleep', 'lost interest', 'crying', 'ሓዘን', 'ጭንቀት'],
            'ptsd': ['ptsd', 'trauma', 'traumatic', 'flashback', 'nightmare', 'nightmares',
                    'war', 'conflict', 'ድሕሪ-ስቓይ', 'ድሕሪ-ጭንቀት'],
            'sexual_trauma': ['rape', 'raped', 'sexual assault', 'sexually assaulted', 'sexual violence',
                             'sexual abuse', 'molested', 'violated', 'forced sex', 'gbv',
                             'gender-based violence', 'touched inappropriately', 'ጾታዊ ዓመጽ',
                             'they raped me', 'soldiers raped', 'gang rape', 'gang raped',
                             'womb is enemy', 'cleanse blood', 'make you give birth'],
            'pregnancy_from_rape': ['pregnant from rape', 'pregnancy from assault', 'pregnancy from rape',
                                   'carrying rapist baby', 'pregnant by soldier', 'pregnant by soldiers',
                                   'don\'t want this baby', 'baby from rape', 'child from assault',
                                   'conceived in rape', 'ጥንሲ ካብ ዓመጽ', 'pregnant from assault'],
            'children_born_of_rape': ['child from rape', 'rape baby', 'enemy child', 'enemy\'s child',
                                     'child looks like rapist', 'child of soldier', 'child of soldiers',
                                     'hate my child', 'love my child but', 'child reminds me',
                                     'see his face in my child', 'ቆልዑ ዝተወለዱ ካብ ዓመጽ'],
            'ethnic_trauma': ['because i am tigrayan', 'because we are tigrayan', 'cleanse blood',
                             'womb is enemy', 'ethnic cleansing', 'they said tigrayans', 'genocide',
                             'destroy tigrayan', 'targeted as tigrayan', 'tigrayan identity',
                             'ንዘርኢ-ጥፍኣት'],
            'hal_circles': ['hal circle', 'peer support', 'survivor group', 'women\'s circle',
                           'healing circle', 'support group for survivors', 'ዓንኬላታት'],
            'anxiety': ['anxious', 'anxiety', 'worried', 'worry', 'panic', 'panic attack',
                       'scared', 'fear', 'racing heart', 'ፍርሒ'],
            'grief': ['grief', 'grieving', 'lost someone', 'death of', 'died', 'mourning',
                     'bereavement', 'miss them', 'ሓዘን'],
            'acute_stress': ['stressed', 'overwhelmed', 'can\'t cope', 'crisis', 'breakdown',
                            'ህጹጽ ጭንቀት'],
            'panic_attack': ['panic attack', 'panicking', 'can\'t breathe', 'heart racing',
                            'ድንገተኛ ፍርሒ'],
            'dissociation': ['flashback', 'dissociating', 'frozen', 'numb', 'not real',
                            'ምፍላይ']
        }

        # Check crisis keywords first (highest priority)
        for condition, keywords in mental_health_crisis_keywords.items():
            if any(kw in text_lower for kw in keywords):
                card_data = self._build_mental_health_card(condition)
                if card_data:
                    cards.append(card_data)
                    return cards  # Return immediately for crisis - don't add other cards

        # Check mental health keywords (before medical to prioritize mental health)
        # Priority order: most specific conditions first, then broader trauma categories
        if domain in ['medical', None, 'general']:
            # Check specific conditions first (pregnancy, children, ethnic, hal circles)
            specific_conditions = ['pregnancy_from_rape', 'children_born_of_rape', 'ethnic_trauma', 'hal_circles']
            matched_specific = False
            for condition in specific_conditions:
                keywords = mental_health_keywords.get(condition, [])
                if any(kw in text_lower for kw in keywords):
                    card_data = self._build_mental_health_card(condition)
                    if card_data:
                        cards.append(card_data)
                        matched_specific = True
                        break

            # If no specific condition matched, check standard conditions
            if not matched_specific:
                for condition, keywords in mental_health_keywords.items():
                    if condition in specific_conditions:
                        continue  # Skip already checked
                    if any(kw in text_lower for kw in keywords):
                        card_data = self._build_mental_health_card(condition)
                        if card_data:
                            cards.append(card_data)
                            break  # One mental health card per query

        # Check medical keywords (only if no mental health card found)
        if domain in ['medical', None, 'general'] and not cards:
            for symptom, keywords in medical_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    card_data = self._build_medical_card(symptom)
                    if card_data:
                        cards.append(card_data)
                        break  # One medical card per query

        # Check agricultural keywords
        if domain in ['agriculture', None, 'general']:
            for topic, keywords in agricultural_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    card_data = self._build_agricultural_card(topic)
                    if card_data:
                        cards.append(card_data)
                        break  # One agricultural card per query

        return cards

    def _build_medical_card(self, symptom):
        """Build a medical Quick Reference card from JSON data"""
        if not self.medical_knowledge_json:
            return None

        # Check symptoms section
        symptoms_data = self.medical_knowledge_json.get('symptoms', {})
        if symptom in symptoms_data:
            data = symptoms_data[symptom]
            return {
                'type': 'medical',
                'keyword': symptom.title(),
                'data': {
                    'danger_signs': data.get('danger_signs', []),
                    'immediate_care': data.get('immediate_care', []),
                    'possible_causes': data.get('possible_causes', [])
                }
            }

        # Check regional diseases
        diseases_data = self.medical_knowledge_json.get('regional_diseases', {})
        if symptom in diseases_data:
            data = diseases_data[symptom]
            return {
                'type': 'medical',
                'keyword': data.get('name', symptom.title()),
                'data': {
                    'danger_signs': data.get('emergency_signs', []),
                    'immediate_care': data.get('treatment', []),
                    'possible_causes': data.get('symptoms', [])
                }
            }

        # Check emergency procedures
        emergency_data = self.medical_knowledge_json.get('emergency_procedures', {})
        for proc_key, proc_data in emergency_data.items():
            if symptom in proc_key:
                return {
                    'type': 'medical',
                    'keyword': proc_key.replace('_', ' ').title(),
                    'data': {
                        'danger_signs': proc_data.get('recognition', []),
                        'immediate_care': proc_data.get('immediate_action', []),
                        'possible_causes': []
                    }
                }

        return None

    def _build_agricultural_card(self, topic):
        """Build an agricultural Quick Reference card from JSON data"""
        if not self.agricultural_knowledge_json:
            return None

        # Check crops section
        crops_data = self.agricultural_knowledge_json.get('crops', {})
        if topic in crops_data:
            data = crops_data[topic]
            return {
                'type': 'agricultural',
                'keyword': data.get('name', topic.title()),
                'tigrinya_name': data.get('tigrinya_name', ''),
                'data': {
                    'planting_time': data.get('planting_time', ''),
                    'harvest_time': data.get('harvest_time', ''),
                    'growing_season': data.get('growing_season', ''),
                    'water_requirements': data.get('water_requirements', ''),
                    'management': data.get('management_practices', data.get('management', [])),
                    'varieties': data.get('varieties', [])
                }
            }

        # Check livestock section
        livestock_data = self.agricultural_knowledge_json.get('livestock', {})
        if topic in livestock_data:
            data = livestock_data[topic]
            mgmt = data.get('management', {})
            return {
                'type': 'agricultural',
                'keyword': topic.title(),
                'tigrinya_name': '',
                'data': {
                    'feeding': mgmt.get('feeding', []),
                    'health': mgmt.get('health', []),
                    'breeding': mgmt.get('breeding', []),
                    'common_diseases': data.get('common_diseases', [])
                }
            }

        # Check for drought in climate adaptation
        if topic == 'drought':
            climate_data = self.agricultural_knowledge_json.get('climate_adaptation', {})
            drought_strategies = climate_data.get('drought_strategies', [])
            water_data = self.agricultural_knowledge_json.get('water_management', {})
            drought_mgmt = water_data.get('drought_management', [])
            return {
                'type': 'agricultural',
                'keyword': 'Drought Management',
                'tigrinya_name': 'ድርቂ',
                'data': {
                    'strategies': drought_strategies + drought_mgmt,
                    'water_harvesting': water_data.get('irrigation', {}).get('water_harvesting', [])
                }
            }

        # Check for planting in calendar
        if topic == 'planting':
            calendar = self.agricultural_knowledge_json.get('traditional_farming_calendar', {})
            return {
                'type': 'agricultural',
                'keyword': 'Planting Calendar',
                'tigrinya_name': 'ምዝራእ',
                'data': {
                    'june': calendar.get('june', ''),
                    'july': calendar.get('july', ''),
                    'february': calendar.get('february', ''),
                    'september': calendar.get('september', '')
                }
            }

        return None

    def _build_mental_health_card(self, condition):
        """Build a mental health Quick Reference card from JSON data"""
        if not self.mental_health_knowledge_json:
            return None

        conditions_data = self.mental_health_knowledge_json.get('conditions', {})

        # PRIORITY 1: Check for specialized Tigray trauma sections FIRST
        # These need custom card structures, not the generic symptoms/danger_signs format

        # Check for pregnancy_from_rape section
        if condition == 'pregnancy_from_rape':
            data = conditions_data.get('pregnancy_from_rape', {})
            if data:
                return {
                    'type': 'mental_health',
                    'keyword': data.get('name', 'Pregnancy from Sexual Violence'),
                    'tigrinya_name': data.get('tigrinya_name', 'ጥንሲ ካብ ዓመጽ'),
                    'is_trauma': True,
                    'data': {
                        'validation': data.get('validation', []),
                        'medical_options': data.get('medical_options', {}),
                        'if_continuing_pregnancy': data.get('if_continuing_pregnancy', []),
                        'stigma_you_may_face': data.get('stigma_you_may_face', []),
                        'support_needs': data.get('support_needs', [])
                    }
                }

        # Check for children_born_of_rape section
        if condition == 'children_born_of_rape':
            data = conditions_data.get('children_born_of_rape', {})
            if data:
                return {
                    'type': 'mental_health',
                    'keyword': data.get('name', 'Children Born of Sexual Violence'),
                    'tigrinya_name': data.get('tigrinya_name', 'ቆልዑ ዝተወለዱ ካብ ዓመጽ'),
                    'is_trauma': True,
                    'data': {
                        'for_mothers': data.get('for_mothers', []),
                        'for_community_helpers': data.get('for_community_helpers', []),
                        'for_the_children': data.get('for_the_children', []),
                        'identity_development': data.get('identity_development', [])
                    }
                }

        # Check for ethnic_trauma - maps to ethnic_identity_healing in sexual_trauma
        if condition == 'ethnic_trauma':
            sexual_trauma_data = conditions_data.get('sexual_trauma', {})
            if sexual_trauma_data:
                return {
                    'type': 'mental_health',
                    'keyword': 'Ethnic Identity Healing',
                    'tigrinya_name': 'ምሕዋይ መንነት',
                    'is_trauma': True,
                    'data': {
                        'ethnic_identity_healing': sexual_trauma_data.get('ethnic_identity_healing', []),
                        'perpetrator_statements': sexual_trauma_data.get('perpetrator_statements_survivors_heard', []),
                        'coping_strategies': sexual_trauma_data.get('coping_strategies', [])[:5],
                        'context': 'Your identity was not destroyed. Your survival resists their attempt to erase Tigrayans.'
                    }
                }

        # Check for hal_circles section
        if condition == 'hal_circles':
            data = conditions_data.get('hal_circles', {})
            if data:
                return {
                    'type': 'mental_health',
                    'keyword': data.get('name', 'HAL Circles (Healing Support)'),
                    'tigrinya_name': data.get('tigrinya_name', 'ዓንኬላታት HAL'),
                    'data': {
                        'how_it_works': data.get('how_it_works', []),
                        'key_principles': data.get('key_principles', []),
                        'what_helps': data.get('what_helps', []),
                        'outcomes': data.get('outcomes', [])
                    }
                }

        # PRIORITY 2: Check conditions section for sexual_trauma and standard conditions
        if condition in conditions_data:
            data = conditions_data[condition]

            # Special handling for sexual_trauma with Tigray-specific content
            if condition == 'sexual_trauma':
                return {
                    'type': 'mental_health',
                    'keyword': data.get('name', 'Genocidal Sexual Violence'),
                    'tigrinya_name': data.get('tigrinya_name', 'ጾታዊ ዓመጽ ንዘርኢ-ጥፍኣት'),
                    'is_trauma': True,
                    'data': {
                        'context': data.get('context', ''),
                        'immediate_support': data.get('immediate_support', [])[:6],
                        'do_not': data.get('do_not', [])[:6],
                        'coping_strategies': data.get('coping_strategies', [])[:5],
                        'ethnic_identity_healing': data.get('ethnic_identity_healing', [])[:4],
                        'medical_timeline': data.get('medical_considerations', {}).get('emergency_timeline', {})
                    }
                }

            # Standard mental health condition card (depression, anxiety, grief, etc.)
            return {
                'type': 'mental_health',
                'keyword': data.get('name', condition.replace('_', ' ').title()),
                'tigrinya_name': data.get('tigrinya_name', ''),
                'data': {
                    'symptoms': data.get('symptoms', []),
                    'danger_signs': data.get('danger_signs', []),
                    'immediate_support': data.get('immediate_support', []),
                    'coping_strategies': data.get('coping_strategies', []),
                    'when_to_refer': data.get('when_to_refer', [])
                }
            }

        # PRIORITY 3: Check crisis intervention section
        crisis_data = self.mental_health_knowledge_json.get('crisis_intervention', {})
        if condition in crisis_data:
            data = crisis_data[condition]
            return {
                'type': 'mental_health',
                'keyword': data.get('name', condition.replace('_', ' ').title()),
                'tigrinya_name': data.get('tigrinya_name', ''),
                'is_crisis': True,
                'data': {
                    'recognition': data.get('recognition', []),
                    'immediate_action': data.get('immediate_action', []),
                    'do_not': data.get('do_not', []),
                    'safety_planning': data.get('safety_planning', [])
                }
            }

        # Check coping strategies section
        coping_data = self.mental_health_knowledge_json.get('coping_strategies', {})
        if condition in coping_data:
            data = coping_data[condition]
            strategies = []
            # Flatten nested coping strategies
            for key, value in data.items():
                if isinstance(value, list):
                    strategies.extend(value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list):
                            strategies.extend(sub_value)
            return {
                'type': 'mental_health',
                'keyword': data.get('name', condition.replace('_', ' ').title()),
                'tigrinya_name': data.get('tigrinya_name', ''),
                'data': {
                    'strategies': strategies[:8]  # Limit to 8 strategies
                }
            }

        return None

    def get_context_stats(self):
        """Get current context usage statistics"""
        if CONTEXT_MANAGER_AVAILABLE:
            cm = get_context_manager()
            return cm.get_context_stats()
        return {'available': False}

    def clear_conversation_history(self):
        """Clear conversation history to start fresh"""
        if CONTEXT_MANAGER_AVAILABLE:
            cm = get_context_manager()
            cm.clear_history()
            return True
        return False

    def set_context_model(self, model):
        """Set the model for context management"""
        if CONTEXT_MANAGER_AVAILABLE:
            cm = get_context_manager()
            cm.set_model(model)

    def _detect_query_domain(self, user_input):
        """Detect the domain of a user query for Wikipedia search."""
        text_lower = user_input.lower()

        # Mental health domain keywords - CHECK FIRST (highest priority for trauma)
        mental_health_keywords = [
            'depressed', 'depression', 'anxiety', 'anxious', 'trauma', 'traumatic',
            'ptsd', 'flashback', 'nightmare', 'suicidal', 'suicide', 'self-harm',
            'kill myself', 'end my life', 'want to die', 'hurt myself',
            'rape', 'raped', 'sexual assault', 'sexual violence', 'molested',
            'pregnant from rape', 'pregnant from assault', 'child from rape', 'violated', 'gbv',
            'panic attack', 'grief', 'grieving', 'hopeless', 'worthless',
            'mental health', 'counseling', 'therapy', 'healing', 'survivor',
            'genocide', 'ethnic cleansing', 'cleanse blood', 'womb is enemy',
            'hal circle', 'support group', 'peer support',
            'ጾታዊ ዓመጽ', 'ጭንቀት', 'ሓዘን', 'ፍርሒ', 'ድሕሪ-ስቓይ'  # Tigrinya mental health words
        ]

        # Medical domain keywords
        medical_keywords = [
            'disease', 'treatment', 'symptom', 'medicine', 'health', 'medical',
            'fever', 'pain', 'headache', 'infection', 'doctor', 'hospital',
            'malaria', 'cholera', 'tuberculosis', 'diarrhea', 'vaccine', 'cure',
            'ሕማም', 'ሓኪም', 'ሕክምና', 'ጥዕና', 'ድውይ'  # Tigrinya medical words
        ]

        # Agricultural domain keywords
        agricultural_keywords = [
            'crop', 'farm', 'plant', 'soil', 'harvest', 'agriculture', 'seed',
            'irrigation', 'drought', 'livestock', 'cattle', 'wheat', 'teff',
            'sorghum', 'barley', 'fertilizer', 'ሕርሻ', 'ዘርኢ', 'ማይ'
        ]

        # Educational domain keywords
        educational_keywords = [
            'what is', 'how does', 'explain', 'learn', 'study', 'science',
            'history', 'math', 'biology', 'chemistry', 'physics', 'geography',
            'education', 'school', 'teach'
        ]

        # African/Arab context keywords
        african_arab_keywords = [
            'ethiopia', 'eritrea', 'somalia', 'sudan', 'egypt', 'morocco',
            'algeria', 'tunisia', 'libya', 'arab', 'arabic', 'islamic',
            'african', 'africa', 'tigray', 'amhara', 'maghreb', 'levant',
            'traditional medicine', 'traditional healing', 'folk medicine',
            'african history', 'arab history', 'middle east', 'sahara',
            'nile', 'horn of africa', 'arabian peninsula'
        ]

        # Count keyword matches - mental health checked FIRST
        mental_health_score = sum(1 for kw in mental_health_keywords if kw in text_lower)
        medical_score = sum(1 for kw in medical_keywords if kw in text_lower)
        agri_score = sum(1 for kw in agricultural_keywords if kw in text_lower)
        edu_score = sum(1 for kw in educational_keywords if kw in text_lower)
        african_arab_score = sum(1 for kw in african_arab_keywords if kw in text_lower)

        # Determine domain with African/Arab context awareness
        # Mental health takes priority for trauma-related queries
        has_african_arab_context = african_arab_score > 0

        # Mental health domain takes priority when trauma keywords detected
        if mental_health_score > 0:
            return 'mental_health'
        elif medical_score >= agri_score and medical_score >= edu_score and medical_score > 0:
            if has_african_arab_context:
                return 'african_arab_medical'
            return 'medical'
        elif agri_score >= medical_score and agri_score >= edu_score and agri_score > 0:
            return 'agriculture'
        elif edu_score > 0:
            if has_african_arab_context:
                return 'african_arab_education'
            return 'education'
        elif has_african_arab_context:
            return 'african_arab_general'
        else:
            return None  # General query

    def _search_wikipedia_context(self, user_input, domain=None):
        """Search Wikipedia for relevant context to enhance AI response."""

        # Enhanced searches disabled for performance - always use standard Wikipedia
        # These non-existent modules were causing 90+ second delays
        # All domains now use consistent, fast Wikipedia search

        # Fallback to general Wikipedia search
        if not self.wikipedia_available or not self.wikipedia_search:
            return None, []

        try:
            # Search for relevant articles
            results = self.wikipedia_search.search(
                user_input,
                max_results=3,
                min_words=100,
                domain=domain
            )

            if not results:
                return None, []

            # Format context for AI model injection
            context = format_for_context(results, max_chars=1500)

            # Return simplified results for UI display
            search_results = [
                {
                    'id': r['id'],  # Article ID for fetching full content
                    'title': r['title'],
                    'summary': r['summary'][:200] + '...' if len(r['summary']) > 200 else r['summary'],
                    'word_count': r['word_count'],
                    'score': r['score']
                }
                for r in results
            ]

            return context, search_results

        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return None, []

    def get_wikipedia_article(self, article_id):
        """Get full Wikipedia article by ID for reading in research panel."""
        if not self.wikipedia_available or not self.wikipedia_search:
            return None

        try:
            # First try main Wikipedia database
            article = self.wikipedia_search.get_article(article_id)
            if article:
                # Content is already cleaned by get_article method
                return {
                    'id': article['id'],
                    'title': article['title'],
                    'content': article['content'],
                    'summary': article.get('summary', ''),
                    'word_count': article['word_count'],
                    'url': article.get('url', '')
                }

            # If not found in main Wikipedia, try medical database
            medical_article = self._get_medical_article(article_id)
            if medical_article:
                return medical_article

            return None
        except Exception as e:
            print(f"Error fetching article {article_id}: {e}")
            return None

    def _get_medical_article(self, article_id):
        """Get article from medical Wikipedia database"""
        conn = None
        try:
            import sqlite3
            from pathlib import Path

            medical_db_path = Path(__file__).parent.parent / "educational_archive" / "knowledge" / "medical_wikipedia.db"

            if not medical_db_path.exists():
                return None

            conn = sqlite3.connect(medical_db_path)
            cursor = conn.cursor()

            cursor.execute(
                'SELECT id, title, content, word_count, url FROM articles WHERE id = ?',
                (article_id,)
            )

            row = cursor.fetchone()

            if row:
                return {
                    'id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'summary': row[2][:500] + "..." if len(row[2]) > 500 else row[2],
                    'word_count': row[3],
                    'url': row[4] if row[4] else f"Medical Article: {row[1]}"
                }
            return None
        except Exception as e:
            print(f"Error fetching medical article {article_id}: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def _query_local_model(self, user_input, model='qwen2.5:0.5b', user_domain=None, english_only=False):
        """Query local AI model with Wikipedia-enhanced context"""
        if not self.local_ai_available:
            return None

        try:
            import requests
            import time

            # Initialize context manager if available
            context_stats = None
            context_warning = None
            if CONTEXT_MANAGER_AVAILABLE:
                cm = get_context_manager()
                cm.set_model(model)
                # Clear context for each query to prevent contamination
                if user_domain:  # Only clear when user explicitly selects domain
                    cm.clear_history()
                    print(f"🧹 Cleared context for fresh {user_domain} domain query")

            # ALWAYS use user's selected domain first, only fallback to detection
            if user_domain and user_domain != 'general':
                domain = user_domain
                print(f"[TARGET] Using user-selected domain: {domain}")
            else:
                domain_start = time.time()
                domain = self._detect_query_domain(user_input)
                domain_time = time.time() - domain_start
                print(f"[SEARCH] Auto-detected domain: {domain} ({domain_time:.3f}s)")

            # Automatic Wikipedia search (optimized FTS5 external content mode - sub-second)
            # CRITICAL: Skip Wikipedia for mental_health domain - trauma queries should NOT
            # have external context injected. This prevents harmful content like historical
            # atrocities being mixed into survivor support responses.
            if domain == 'mental_health':
                wiki_context, wiki_results = None, []
                print("[MENTAL HEALTH] Wikipedia search DISABLED for trauma-informed response")
            else:
                wiki_context, wiki_results = self._search_wikipedia_context(user_input, domain=domain)
                if wiki_results:
                    print(f"[WIKI] Wikipedia: {len(wiki_results)} articles found")

            print("[LAUNCH] Proceeding to AI response")

            # Build enhanced system prompt based on domain
            if domain == 'mental_health':
                base_prompt = (
                    "You are NATNA AI, a trauma-informed mental health support assistant created by NATNA Children's Foundation. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created to provide compassionate mental health support. "
                    "You have comprehensive clinical knowledge but deliver it with warmth and compassion.\n\n"

                    "CRITICAL GUIDELINES FOR SEXUAL VIOLENCE SUPPORT:\n"
                    "- ALWAYS believe the survivor. Say 'I believe you.'\n"
                    "- NEVER ask for details of the assault\n"
                    "- Validate feelings: 'What you feel is normal after what happened'\n"
                    "- Recognize the genocidal dimension: survivors in Tigray were targeted AS Tigrayans\n"
                    "- Empower choice: 'You decide what is right for you'\n"
                    "- Pace information: do not overwhelm - offer to share more if they want\n"
                    "- Normalize seeking help: 'Reaching out takes courage'\n\n"

                    "KEY MESSAGES TO ALWAYS INCLUDE:\n"
                    "- 'This was not your fault'\n"
                    "- 'You are not alone - many survivors share these feelings'\n"
                    "- 'Healing is possible, in your own time'\n"
                    "- 'Your identity was not destroyed by what happened'\n"
                    "- 'Your survival is resistance'\n\n"

                    "FOR PREGNANCY FROM RAPE:\n"
                    "- Whatever she decides is HER choice\n"
                    "- Complex feelings are normal\n"
                    "- Do not pressure toward any decision\n"
                    "- Children are innocent of how they were conceived\n\n"

                    "AVOID:\n"
                    "- Clinical coldness\n"
                    "- Pressuring for details\n"
                    "- Suggesting forgiveness before they are ready\n"
                    "- Minimizing or comparing traumas\n"
                    "- Asking 'why didn't you fight back'\n"
                    "- Using words like 'victim' unless they use it first - prefer 'survivor'\n\n"

                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis. Write plain text only. No markdown formatting. "
                    "RESPONSE STYLE: Be warm and compassionate, not clinical. Do NOT show your thinking process. "
                    "Give supportive, validating responses that center the survivor's experience and choices."
                )
            elif domain == 'medical':
                base_prompt = (
                    "I am NATNA AI, developed by NATNA Children's Foundation. I am not Qwen, Claude, GPT, or any other AI - I am NATNA AI. "
                    "I must always identify myself as NATNA AI in responses. "
                    "You are NATNA AI, a medical assistant created by NATNA Children's Foundation. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide medical support. "
                    "Provide helpful medical advice in both English and Tigrinya. "
                    "Focus on practical healthcare guidance. Always recommend seeing a doctor for serious conditions. "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis. Write plain text only. No markdown formatting. "
                    "Keep responses concise and helpful."
                )
            elif domain == 'agriculture':
                base_prompt = (
                    "You are NATNA AI, an agricultural assistant created by NATNA Children's Foundation. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide agricultural support. "
                    "Provide practical farming advice considering drought conditions and local crops for Tigray region farmers. "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis. Write plain text only. No markdown formatting. "
                    "RESPONSE STYLE: Be direct and professional like an expert agricultural advisor. Do NOT show your thinking process or say things like 'Let me think', 'I remember learning', 'Wait, let me recall', or 'I'm trying to understand'. "
                    "Give practical, confident farming guidance without narrating your thought process. Sound like a knowledgeable agricultural expert, not a child thinking out loud."
                )
            elif domain == 'african_arab_medical':
                base_prompt = (
                    "You are NATNA AI, a medical assistant created by NATNA Children's Foundation with expertise in African and Arab world healthcare. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide medical support for African and Arab communities. "
                    "Provide medical advice combining traditional African and Arab medicine with modern healthcare practices. Always recommend consulting healthcare providers for serious conditions. "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis. Write plain text only. No markdown formatting. "
                    "RESPONSE STYLE: Be direct and professional like a knowledgeable healthcare advisor. Do NOT show your thinking process or say things like 'Let me think', 'I remember learning', 'Wait, let me recall', or 'I'm trying to understand'. "
                    "Give clear, confident medical guidance without narrating your thought process. Sound like an experienced medical expert, not a child thinking out loud."
                )
            elif domain == 'african_arab_education':
                base_prompt = (
                    "You are NATNA AI, an educational assistant created by NATNA Children's Foundation with expertise in African and Arab world studies. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide educational support about African and Arab history, culture, and geography. "
                    "Provide comprehensive explanations about African and Arab civilizations, history, geography, languages, and cultures for K-12 and college students. "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis. Write plain text only. No markdown formatting. "
                    "RESPONSE STYLE: Be direct and professional like an expert scholar. Do NOT show your thinking process or say things like 'Let me think', 'I remember learning', 'Wait, let me recall', or 'I'm trying to understand'. "
                    "Give authoritative, confident explanations without narrating your thought process. Sound like a knowledgeable historian and cultural expert, not a child thinking out loud."
                )
            elif domain == 'african_arab_general':
                base_prompt = (
                    "You are NATNA AI, a knowledge assistant created by NATNA Children's Foundation with expertise in African and Arab world topics. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide comprehensive information about Africa and the Arab world. "
                    "Provide accurate, culturally sensitive information about African and Arab countries, peoples, cultures, languages, and current affairs. Be respectful of cultural diversity. "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis. Write plain text only. No markdown formatting. "
                    "RESPONSE STYLE: Be direct and professional like a knowledgeable expert. Do NOT show your thinking process or say things like 'Let me think', 'I remember learning', 'Wait, let me recall', or 'I'm trying to understand'. "
                    "Give clear, confident information without narrating your thought process. Sound like an expert on African and Arab world topics, not a child thinking out loud."
                )
            elif domain == 'education':
                base_prompt = (
                    "You are NATNA AI, a K-12 educational tutor created by NATNA Children's Foundation. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide K-12 educational support. "
                    "Provide clear explanations for math, science, history, language arts, and other K-12 subjects appropriate for the student's grade level. "
                    "Only provide educational explanations when asked specific academic questions. Respond naturally to historical figures or general topics. "
                    "MATH FORMATTING: When writing mathematical expressions, use LaTeX notation with $$ delimiters for display equations (e.g., $$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$) and \\( \\) for inline math (e.g., \\(x^2 + y^2 = r^2\\)). "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis in regular text. Write plain text for non-math content. "
                    "RESPONSE STYLE: Be direct and professional like a knowledgeable teacher. Do NOT show your thinking process or say things like 'Let me think', 'I remember learning', 'Wait, let me recall', or 'I'm trying to understand'. "
                    "Give concise, confident explanations without narrating your thought process. Sound like an expert educator, not a child thinking out loud."
                )
            elif domain == 'college':
                base_prompt = (
                    "You are NATNA AI, a college-level educational assistant created by NATNA Children's Foundation. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide college-level educational support. "
                    "Provide detailed, advanced explanations about higher education topics including advanced mathematics, sciences, research methods, and academic subjects for Tigrinya-speaking communities. "
                    "Focus on in-depth analysis suitable for college and graduate students. Include theoretical frameworks, research citations when relevant, and complex problem-solving approaches. "
                    "MATH FORMATTING: When writing mathematical expressions, use LaTeX notation with $$ delimiters for display equations (e.g., $$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$) and \\( \\) for inline math (e.g., \\(e^{i\\pi} + 1 = 0\\)). Always format equations, formulas, and mathematical symbols using LaTeX. "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis in regular text. Write plain text for non-math content. "
                    "Structure your response as follows: First, provide comprehensive analysis and explanation. "
                    "Then, if additional context is provided below, add a 'Further Context' section that incorporates any relevant supplementary information. "
                    "RESPONSE STYLE: Be direct and scholarly. Do NOT show your thinking process or say things like 'Let me think', 'I remember learning', 'Wait, let me recall', or 'I'm trying to understand'. "
                    "Give authoritative, confident explanations without narrating your thought process. Sound like a knowledgeable professor, not a student thinking out loud."
                )
            elif domain == 'technical':
                base_prompt = (
                    "You are NATNA AI, a technical assistant created by NATNA Children's Foundation. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide technical support. "
                    "Provide detailed technical explanations about engineering, technology, and scientific processes for Tigrinya-speaking communities. "
                    "Focus on practical technical knowledge and problem-solving. "
                    "MATH FORMATTING: When writing formulas or equations, use LaTeX notation with $$ delimiters for display equations and \\( \\) for inline math. "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis in regular text. Write plain text for non-math content. "
                    "RESPONSE STYLE: Be direct and professional like a technical expert. Do NOT show your thinking process or say things like 'Let me think', 'I remember learning', 'Wait, let me recall', or 'I'm trying to understand'. "
                    "Give precise, confident technical explanations without narrating your thought process. Sound like a knowledgeable engineer, not a child thinking out loud."
                )
            elif domain == 'programming':
                base_prompt = (
                    "You are NATNA AI, a programming assistant created by NATNA Children's Foundation. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide programming support. "
                    "Provide clear explanations about programming concepts, coding best practices, and software development for Tigrinya-speaking communities. "
                    "Include code examples when helpful. "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis. Write plain text only. No markdown formatting except for code blocks. "
                    "RESPONSE STYLE: Be direct and professional like a programming expert. Do NOT show your thinking process or say things like 'Let me think', 'I remember learning', 'Wait, let me recall', or 'I'm trying to understand'. "
                    "Give precise, confident code explanations without narrating your thought process. Sound like a skilled software engineer, not a child thinking out loud."
                )
            elif domain == 'general':
                base_prompt = (
                    "You are NATNA AI, a general knowledge assistant created by NATNA Children's Foundation. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide general knowledge support. "
                    "Provide clear, accurate information on a wide range of topics for Tigrinya-speaking communities. Be careful about accuracy and say so if you're uncertain about facts. "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis. Write plain text only. No markdown formatting. "
                    "RESPONSE STYLE: Be direct and professional. Do NOT show your thinking process or say things like 'Let me think', 'I remember learning', 'Wait, let me recall', or 'I'm trying to understand'. "
                    "Give concise, confident answers without narrating your thought process. Sound like a knowledgeable expert, not a child thinking out loud."
                )
            else:
                # Fallback for undefined domains
                base_prompt = (
                    "You are NATNA AI, an assistant created by NATNA Children's Foundation. "
                    "Your name is NATNA AI, not Qwen or any other name. You were created by NATNA Children's Foundation to provide support. "
                    "Provide clear, accurate, and helpful responses for Tigrinya-speaking communities. "
                    "TEXT FORMATTING: Do NOT use asterisks (*) or underscores (_) for emphasis. Write plain text only. No markdown formatting. "
                    "RESPONSE STYLE: Be direct and professional. Do NOT show your thinking process or say things like 'Let me think', 'I remember learning', 'Wait, let me recall', or 'I'm trying to understand'. "
                    "Give concise, confident answers without narrating your thought process. Sound like a knowledgeable expert, not a child thinking out loud."
                )

            # Build structured messages for /api/chat endpoint
            if CONTEXT_MANAGER_AVAILABLE:
                if wiki_context:
                    cm.set_wikipedia_context(wiki_context)
                messages, context_stats, truncated = cm.prepare_messages(user_input, base_prompt)
                if truncated:
                    context_warning = f"Context was truncated to fit model limit ({get_model_limit(model)} tokens)"
            else:
                # Simple fallback — still structured messages for /api/chat
                system_content = base_prompt
                if wiki_context:
                    system_content += f"\n\nReference Information:\n{wiki_context}"
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input}
                ]

            # Use the selected model (maps to local Ollama model names)
            ollama_model = model  # Model names match our manifest names

            # Debug: show what we're sending to Ollama
            print(f"[DEBUG] Sending {len(messages)} messages to Ollama:")
            for i, msg in enumerate(messages):
                role = msg.get('role', '?')
                content_preview = msg.get('content', '')[:120]
                print(f"  [{i}] {role}: {content_preview}...")

            # Use local Ollama /api/chat with structured messages
            ollama_start = time.time()
            print(f"[TIME] Starting Ollama API call to {ollama_model} at {ollama_start:.3f}")
            response = requests.post('http://localhost:11434/api/chat',
                json={
                    'model': ollama_model,
                    'messages': messages,
                    'stream': False,
                    'options': {
                        'num_predict': 2048,
                        'temperature': 0.7,
                    }
                },
                timeout=120 if '16gb' in model else 60  # Longer timeout for larger model
            )
            ollama_time = time.time() - ollama_start
            print(f"[TIME] Ollama API call completed in {ollama_time:.3f} seconds")

            if response.status_code == 200:
                ai_response = response.json().get('message', {}).get('content', '').strip()

                # Strip qwen3 thinking blocks (activated by /api/chat format)
                ai_response = re.sub(r'<think>.*?</think>', '', ai_response, flags=re.DOTALL).strip()

                if ai_response:
                    # Save both messages to history AFTER successful response
                    if CONTEXT_MANAGER_AVAILABLE:
                        cm.add_user_message(user_input)
                        cm.add_assistant_message(ai_response)
                        cm.clear_wikipedia_context()  # Clear after use

                    # Clean response without messy progressive loading messages
                    result = {
                        'english': ai_response,
                        'tigrinya': None if english_only else self._translate_to_tigrinya_simple(ai_response),
                        'confidence': 0.95,
                        'source': 'local_ai_with_wikipedia' if wiki_results else 'local_ai'
                    }

                    # Add Wikipedia sources if used
                    if wiki_results:
                        result['wikipedia_sources'] = wiki_results
                        result['source'] = 'local_ai_with_wikipedia'
                        result['domain'] = domain

                    # Add context stats if available
                    if context_stats:
                        result['context_stats'] = context_stats
                    if context_warning:
                        result['context_warning'] = context_warning

                    # Clean simple approach: Wikipedia sources available for "Expand Answer" button
                    print(f"[OK] Response ready with {len(wiki_results)} Wikipedia sources for expansion")

                    return result

        except Exception as e:
            print(f"Local AI query failed: {e}")
            return None

    def _translate_to_tigrinya_simple(self, english_text):
        """Quick translation helper using common medical phrases"""
        translations = {
            'rest': 'ዕረፍት ውሰድ',
            'drink water': 'ማይ ስተ',
            'headache': 'ሕማም ርእሲ',
            'see a doctor': 'ሓኪም ረኣይ',
            'take medicine': 'ሕክምና ውሰድ',
            'pain': 'ሕማም',
            'fever': 'ሓሙሽሽ'
        }

        result = english_text.lower()
        for eng, tig in translations.items():
            result = result.replace(eng, tig)
        return result

    def get_connection(self):
        """Get optimized SQLite connection with dynamic memory allocation"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)

        # Calculate dynamic memory allocation based on available system resources
        try:
            virtual_memory = psutil.virtual_memory()
            total_ram_gb = virtual_memory.total / (1024**3)

            # Use 25% of total RAM for database memory mapping (maximum resource utilization)
            db_memory_bytes = int(total_ram_gb * 1024**3 * 0.25)

            # Ensure minimum 256MB and maximum 8GB
            db_memory_bytes = max(268435456, min(db_memory_bytes, 8589934592))

        except (OSError, AttributeError):
            # Fallback to 1GB if system detection fails
            db_memory_bytes = 1073741824

        # Maximum performance SQLite optimizations
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 50000")  # Increased cache size
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute(f"PRAGMA mmap_size = {db_memory_bytes}")  # Dynamic memory map

        print(f"[FIRE] Database memory: {db_memory_bytes // 1024 // 1024}MB")

        return conn

    def analyze_phrase(self, tigrinya_text):
        """Analyze Tigrinya phrase and extract meaning"""
        words = re.findall(r'[\u1200-\u137F]+', tigrinya_text)

        conn = self.get_connection()
        cursor = conn.cursor()

        # Look up each word
        word_meanings = {}
        domains_found = set()

        for word in words:
            cursor.execute('''
                SELECT english, domain, confidence FROM translations
                WHERE tigrinya = ? AND english IS NOT NULL
                ORDER BY confidence DESC LIMIT 1
            ''', (word,))

            result = cursor.fetchone()
            if result:
                english, domain, confidence = result
                word_meanings[word] = {
                    'english': english,
                    'domain': domain,
                    'confidence': confidence
                }
                domains_found.add(domain)

        conn.close()

        return {
            'words': words,
            'meanings': word_meanings,
            'domains': list(domains_found),
            'translated_words': [word_meanings.get(w, {}).get('english', w) for w in words]
        }

    def understand_intent(self, analysis):
        """Understand what the user is asking - Enhanced with context understanding"""
        english_words = analysis['translated_words']
        english_text = ' '.join(english_words).lower()

        # Get original Tigrinya words for better pattern matching
        tigrinya_words = analysis['words']
        tigrinya_text = ' '.join(tigrinya_words)

        # Enhanced medical intent patterns
        # Check for medical keywords in Tigrinya first
        medical_tigrinya = ['ሕማም', 'ሓኪም', 'ሕክምና', 'ጥዕና', 'ድውይ', 'ርእሲ', 'ሕማቅ']
        has_medical_tigrinya = any(word in tigrinya_text for word in medical_tigrinya)

        # Check for medical keywords in English translation
        medical_english = ['disease', 'head', 'pain', 'doctor', 'medicine', 'health', 'treat', 'cure', 'sick']
        has_medical_english = any(word in english_text for word in medical_english)

        # Check domains for medical content
        has_medical_domain = any(domain in ['health', 'medical'] for domain in analysis['domains'])

        # Headache specific patterns
        headache_patterns = {
            'tigrinya': ['ሕማም ርእሲ', 'ርእሲ ሕማም'],
            'english': ['head disease', 'disease head', 'headache']
        }

        is_headache_query = (
            any(pattern in tigrinya_text for pattern in headache_patterns['tigrinya']) or
            any(pattern in english_text for pattern in headache_patterns['english'])
        )

        # Question patterns
        question_patterns = ['እንታይ', 'ከመይ', 'what', 'how', 'የሕክም', 'treats']
        is_question = any(pattern in tigrinya_text or pattern in english_text for pattern in question_patterns)

        # Medical query detection
        if (has_medical_tigrinya or has_medical_english or has_medical_domain) and is_question:
            if is_headache_query:
                return {
                    'intent': 'medical_query',
                    'type': 'headache_treatment',
                    'confidence': 0.9
                }
            else:
                return {
                    'intent': 'medical_query',
                    'type': 'general_health',
                    'confidence': 0.8
                }

        # Agricultural intent
        if any(domain in ['agriculture'] for domain in analysis['domains']):
            return {
                'intent': 'agricultural_query',
                'type': 'farming',
                'confidence': 0.8
            }

        # Greeting intent
        if any(word in english_text for word in ['hello', 'hi', 'greetings']):
            return {
                'intent': 'greeting',
                'type': 'hello',
                'confidence': 0.9
            }

        return {
            'intent': 'general',
            'type': 'unknown',
            'confidence': 0.5
        }

    def generate_response(self, intent, analysis):
        """Generate appropriate response in both languages"""
        if intent['intent'] == 'medical_query':
            if intent['type'] == 'headache_treatment':
                return self.get_medical_response('headache')
            else:
                return self.get_medical_response('general')

        elif intent['intent'] == 'agricultural_query':
            return self.get_agricultural_response()

        elif intent['intent'] == 'greeting':
            return self.get_greeting_response()

        else:
            return self.get_general_response(analysis)

    def get_medical_response(self, condition):
        """Get medical response from database"""
        if condition == 'headache':
            # Build response using database words
            response_en = "For headaches, try: rest, drink water, take medicine if needed. See doctor if severe."
            response_ti = self.translate_to_tigrinya(response_en)

            return {
                'english': response_en,
                'tigrinya': response_ti,
                'confidence': 0.9,
                'source': 'medical_knowledge'
            }

        return {
            'english': "Please consult a doctor for health concerns.",
            'tigrinya': "ሓኪም ወዲ ተወከስ።",
            'confidence': 0.8,
            'source': 'medical_knowledge'
        }

    def translate_to_tigrinya(self, english_text):
        """Translate English response to Tigrinya using database"""
        words = english_text.lower().split()

        conn = self.get_connection()
        cursor = conn.cursor()

        tigrinya_words = []

        for word in words:
            # Clean word
            clean_word = re.sub(r'[^\w]', '', word)
            if not clean_word:
                continue

            # Look up in database
            cursor.execute('''
                SELECT tigrinya FROM translations
                WHERE english = ? AND confidence > 0.5
                ORDER BY confidence DESC LIMIT 1
            ''', (clean_word,))

            result = cursor.fetchone()
            if result:
                tigrinya_words.append(result[0])
            else:
                # Keep original if no translation
                tigrinya_words.append(clean_word)

        conn.close()

        return ' '.join(tigrinya_words)

    def get_agricultural_response(self):
        return {
            'english': "For farming questions, consider soil, water, and proper crop rotation.",
            'tigrinya': "ንሕርሻ ሕቶታት፡ ሓመድ፡ ማይ፡ ከምኡ'ውን ግቡእ ምልውዋጥ ዝርእቲ ኣብ ግምት ኣእቱ።",
            'confidence': 0.8,
            'source': 'agricultural_knowledge'
        }

    def get_greeting_response(self):
        return {
            'english': "Hello! How can I help you today?",
            'tigrinya': "ሰላም! ሎሚ ከመይ ክሕግዘካ እኽእል?",
            'confidence': 0.9,
            'source': 'greeting'
        }

    def get_general_response(self, analysis):
        words = analysis['words']

        # Check if this is English input (no Tigrinya Unicode characters)
        is_english_input = all(not ('\u1200' <= char <= '\u137F') for word in words for char in word)

        if is_english_input and words:
            # Handle English input - provide helpful response
            original_text = ' '.join(words)

            # Check for medical keywords in English
            medical_keywords = ['headache', 'pain', 'sick', 'medicine', 'doctor', 'health', 'treatment']
            if any(keyword in original_text.lower() for keyword in medical_keywords):
                return {
                    'english': "I can help with medical questions. For specific symptoms, please ask in more detail or in Tigrinya for more accurate responses.",
                    'tigrinya': "ብሕክምናዊ ሕቶታት ክሕግዘካ እኽእል። ንፍሉይ ምልክታት፡ ዝያዳ ብዝርዝር ወይ ብትግርኛ ሕተት።",
                    'confidence': 0.7,
                    'source': 'english_medical_general'
                }
            else:
                return {
                    'english': f"Hello! I understand your message: '{original_text}'. I'm a medical assistant that works best with Tigrinya questions. How can I help you today?",
                    'tigrinya': "ሰላም! ናይ ሕክምና ሓላፊ ኢየ። ብትግርኛ ሕቶታት ክመልስ እኽእል። ሎሚ ከመይ ክሕግዘካ?",
                    'confidence': 0.7,
                    'source': 'english_general'
                }
        else:
            # Original Tigrinya processing
            english_words = ' '.join(analysis['translated_words'])
            if english_words.strip():
                return {
                    'english': f"I understand: {english_words}. How can I help?",
                    'tigrinya': f"ተረዲአኒ: {' '.join(analysis['words'])}። ከመይ ክሕግዘካ?",
                    'confidence': 0.6,
                    'source': 'general'
                }
            else:
                return {
                    'english': "I understand your Tigrinya message. Please ask medical questions for the best help.",
                    'tigrinya': "ናይ ትግርኛ መልእኽቲ ተረዲአኒ። ሕክምናዊ ሕቶታት ሓቶ።",
                    'confidence': 0.5,
                    'source': 'general'
                }

    def process_query(self, tigrinya_text, model='qwen2.5:0.5b', domain=None, english_only=False):
        """Full pipeline: analyze -> understand -> respond with advanced capabilities"""

        # Try local AI first for all queries
        if self.use_local_ai and self.local_ai_available:
            ai_result = self._query_local_model(tigrinya_text, model=model, user_domain=domain, english_only=english_only)
            if ai_result:
                return {'response': ai_result}

        # Fallback to SQL + pattern matching
        # Check if this needs advanced processing (complex English questions)
        is_complex_question = len(tigrinya_text.split()) > 3 and not any('\u1200' <= char <= '\u137F' for char in tigrinya_text)

        if is_complex_question:
            # Use advanced analysis for complex questions
            complex_analysis = self.analyze_complex_question(tigrinya_text)

            if complex_analysis['type'] == 'agricultural':
                response = self.generate_advanced_agricultural_response(complex_analysis)
            elif complex_analysis['type'] == 'medical':
                response = self.generate_advanced_medical_response(complex_analysis)
            else:
                # Fall back to basic processing
                analysis = self.analyze_phrase(tigrinya_text)
                intent = self.understand_intent(analysis)
                response = self.generate_response(intent, analysis)
                complex_analysis = None

            if complex_analysis:
                return {
                    'input': tigrinya_text,
                    'analysis': complex_analysis,
                    'intent': {
                        'intent': complex_analysis['type'],
                        'type': complex_analysis['type'],
                        'confidence': response['confidence']
                    },
                    'response': response,
                    'debug': {
                        'words_found': len(complex_analysis['entities']),
                        'domains': [complex_analysis['type']],
                        'english_interpretation': tigrinya_text,
                        'entities': complex_analysis['entities']
                    }
                }

        # Standard processing for Tigrinya text and simple questions
        analysis = self.analyze_phrase(tigrinya_text)
        intent = self.understand_intent(analysis)
        response = self.generate_response(intent, analysis)

        return {
            'input': tigrinya_text,
            'analysis': analysis,
            'intent': intent,
            'response': response,
            'debug': {
                'words_found': len(analysis['meanings']),
                'domains': analysis['domains'],
                'english_interpretation': ' '.join(analysis['translated_words'])
            }
        }

    def load_medical_knowledge(self):
        """Load comprehensive medical knowledge base"""
        return {
            'headache': {
                'symptoms': ['pain', 'pressure', 'throbbing'],
                'treatments': [
                    'Rest in a quiet, dark room',
                    'Drink plenty of water to stay hydrated',
                    'Apply cold or warm compress to head/neck',
                    'Take over-the-counter pain relievers if needed',
                    'Practice deep breathing or meditation'
                ],
                'when_to_see_doctor': [
                    'Sudden, severe headache unlike any before',
                    'Headache with fever, stiff neck, confusion',
                    'Headache after head injury',
                    'Headache that gets progressively worse'
                ],
                'tigrinya_advice': 'ንሕማም ርእሲ፡ ዕረፍቲ፡ ማይ ስተ፡ ቀዲሑ ወይ ውዑይ ኮምፕረስ ሰዓበት።'
            },
            'fever': {
                'treatments': ['rest', 'hydration', 'fever reducers', 'cool cloths'],
                'tigrinya_advice': 'ንትኽሳስ፡ ዕረፍቲ፡ ማይ ስተ፡ ቀዳዲ መመሊስ መድሃኒት።'
            }
        }

    def load_agricultural_knowledge(self):
        """Load comprehensive agricultural knowledge for Tigray region"""
        return {
            'sesame': {
                'best_regions': ['Western Tigray', 'Central Tigray', 'Southern Tigray'],
                'drought_strategies': [
                    'Plant drought-resistant varieties like Setit-1',
                    'Use water harvesting (terracing, bunds)',
                    'Plant early in rainy season',
                    'Apply organic mulch to retain moisture',
                    'Practice intercropping with drought-tolerant crops'
                ],
                'tigrinya_advice': 'ሰሳም ብደረቅቲ ዓይነት ክትዘሕብ ትኽእል። ማይ ንምቅማጥ ተራንስ፡ ብእዋኑ ክትዘሕብ።'
            },
            'sorghum': {
                'drought_tolerance': 'Highly drought tolerant',
                'best_varieties': ['Meko-1', 'Gobiye', 'Degalit'],
                'tigrinya_advice': 'ማሳጋ ንደረቅቲ ጽዑቕ፡ ኣብ ደረቅ ከባቢ ክትዘሕቦ ትኽእል።'
            }
        }

    def load_regional_knowledge(self):
        """Load knowledge about different regions of Tigray"""
        return {
            'western_tigray': {
                'climate': 'Semi-arid with moderate rainfall',
                'main_crops': ['sesame', 'sorghum', 'cotton'],
                'challenges': ['drought', 'soil degradation']
            },
            'central_tigray': {
                'climate': 'Highland climate with better rainfall',
                'main_crops': ['teff', 'barley', 'wheat'],
                'challenges': ['erosion', 'small plot sizes']
            }
        }

    def analyze_complex_question(self, text):
        """Analyze questions to understand intent and extract entities"""
        text_lower = text.lower()

        entities = {
            'crops': [],
            'regions': [],
            'conditions': [],
            'medical_terms': [],
            'actions': []
        }

        # Crop detection
        crops = ['sesame', 'sorghum', 'teff', 'barley', 'wheat', 'maize']
        for crop in crops:
            if crop in text_lower:
                entities['crops'].append(crop)

        # Region detection
        regions = ['tigray', 'western tigray', 'central tigray', 'southern tigray']
        for region in regions:
            if region in text_lower:
                entities['regions'].append(region)

        # Condition detection
        conditions = ['drought', 'rain', 'dry', 'wet', 'erosion']
        for condition in conditions:
            if condition in text_lower:
                entities['conditions'].append(condition)

        # Medical terms
        medical_terms = ['headache', 'fever', 'pain', 'sick', 'medicine', 'doctor']
        for term in medical_terms:
            if term in text_lower:
                entities['medical_terms'].append(term)

        # Determine question type
        if entities['medical_terms']:
            question_type = 'medical'
        elif entities['crops'] or 'farm' in text_lower:
            question_type = 'agricultural'
        else:
            question_type = 'general'

        return {
            'type': question_type,
            'entities': entities,
            'original_text': text
        }

    def generate_advanced_agricultural_response(self, analysis):
        """Generate detailed agricultural advice"""
        entities = analysis['entities']

        if 'sesame' in entities['crops'] and 'drought' in entities['conditions']:
            sesame_info = self.agricultural_knowledge['sesame']

            response = "[AGRI] **Sesame Farming During Drought in Tigray:**\n\n"
            response += "**Drought-Resistant Strategies:**\n"
            for strategy in sesame_info['drought_strategies']:
                response += f"• {strategy}\n"

            response += "\n**Regional Advice:** Western and Southern Tigray are best for drought-resistant sesame.\n"

            return {
                'english': response,
                'tigrinya': sesame_info['tigrinya_advice'] + " ኣብ ደረቅቲ እዋን፡ ፍሉይ ዘይነት ተጠቐም።",
                'confidence': 0.9,
                'source': 'advanced_agricultural'
            }

        # General agricultural advice
        return {
            'english': "[AGRI] I can help with agricultural questions. Please specify the crop, region, and conditions.",
            'tigrinya': "ብሕርሻ ክሕግዘካ እኽእል። እንታይ ዝርኢ፡ ኣበይ ዞባ፡ እንታይ ኩነታት እዩ ሓብሮ።",
            'confidence': 0.7,
            'source': 'agricultural_general'
        }

    def generate_advanced_medical_response(self, analysis):
        """Generate detailed medical advice"""
        entities = analysis['entities']

        if 'headache' in entities['medical_terms']:
            headache_info = self.medical_knowledge['headache']

            response = "[MED] **Headache Treatment:**\n\n"
            response += "**Immediate Relief:**\n"
            for treatment in headache_info['treatments']:
                response += f"• {treatment}\n"

            response += "\n**When to See a Doctor:**\n"
            for warning in headache_info['when_to_see_doctor']:
                response += f"• {warning}\n"

            return {
                'english': response,
                'tigrinya': headache_info['tigrinya_advice'],
                'confidence': 0.9,
                'source': 'advanced_medical'
            }

        return {
            'english': "[MED] I can provide medical guidance. Please describe your symptoms.",
            'tigrinya': "ሕክምናዊ ምኽሪ ክህብ እኽእል። ምልክታትካ ግለጽ።",
            'confidence': 0.7,
            'source': 'medical_general'
        }

    def load_medical_responses(self):
        """Load medical response templates"""
        return {
            'headache': {
                'treatments': ['rest', 'water', 'medicine'],
                'advice': 'see doctor if severe'
            }
        }

    def load_agricultural_responses(self):
        """Load agricultural response templates"""
        return {
            'general': {
                'topics': ['soil', 'water', 'crops'],
                'advice': 'consider proper rotation'
            }
        }

    def get_stats(self):
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM translations')
        total = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM translations WHERE english IS NOT NULL')
        translated = cursor.fetchone()[0]

        cursor.execute('SELECT domain, COUNT(*) FROM translations GROUP BY domain')
        domains = dict(cursor.fetchall())

        conn.close()

        return {
            'total_words': total,
            'translated_words': translated,
            'coverage': f"{translated/total*100:.1f}%",
            'domains': domains
        }

    # Interface compatibility methods
    def get_loading_status(self):
        """Compatibility with web interface"""
        wiki_status = f" + {self.wikipedia_search.stats.get('quality_articles', 0):,} Wikipedia articles" if self.wikipedia_available else ""
        if self.local_ai_available:
            return {"status": "ready", "message": f"Local AI + SQL ready with 168K translations{wiki_status}"}
        else:
            return {"status": "ready", "message": f"SQL pattern matching ready with 168K translations{wiki_status}"}

    def translate_response(self, text):
        """Main method used by web interface"""
        if not text.strip():
            return {"english": "", "tigrinya": "", "amharic": "", "confidence": "low", "source_language": "unknown"}

        # Use intelligent processing
        result = self.process_query(text.strip())

        # Determine source language
        is_tigrinya = bool(re.search(r'[\u1200-\u137F]', text))
        source_lang = "tigrinya" if is_tigrinya else "english"

        if is_tigrinya:
            # Tigrinya input -> English response
            return {
                "english": result['response']['english'],
                "tigrinya": text,
                "amharic": "",  # Not supported
                "confidence": "high",
                "source_language": "tigrinya"
            }
        else:
            # English input -> Tigrinya response
            tigrinya_response = self.translate_to_tigrinya(text)
            return {
                "english": text,
                "tigrinya": tigrinya_response,
                "amharic": "",  # Not supported
                "confidence": "high",
                "source_language": "english"
            }


    def detect_language(self, text):
        """Detect language for compatibility"""
        return "tigrinya" if re.search(r'[\u1200-\u137F]', text) else "english"

    def get_followup_response(self, followup_id):
        """Get cached follow-up response by ID"""
        return self.followup_cache.get(followup_id)

    def clear_followup_cache(self):
        """Clear all cached follow-up responses (called on shutdown)"""
        count = len(self.followup_cache)
        self.followup_cache.clear()
        print(f"🧹 Cleared {count} cached follow-up responses")

    def _cleanup_old_cache_entries(self):
        """Remove old cache entries to prevent memory overflow"""
        if len(self.followup_cache) > self.max_cache_size:
            # Remove oldest entries (by timestamp)
            sorted_items = sorted(self.followup_cache.items(), key=lambda x: x[1]['timestamp'])
            to_remove = len(self.followup_cache) - self.max_cache_size + 10  # Remove extra to avoid frequent cleanup

            for i in range(to_remove):
                del self.followup_cache[sorted_items[i][0]]

            print(f"🧹 Removed {to_remove} old cache entries (cache size: {len(self.followup_cache)})")

def main():
    """Test the intelligent translator"""
    translator = IntelligentTigrinyaTranslator()

    print("[BRAIN] Intelligent Tigrinya Translator Test")
    print("=" * 50)

    # Show stats
    stats = translator.get_stats()
    print(f"[STATS] Database: {stats['translated_words']:,} translated / {stats['total_words']:,} total ({stats['coverage']})")

    # Test the medical question
    test_queries = [
        "ሕማም ርእሲ እንታይ የሕክም?",  # What treats headaches?
        "ሰላም",                    # Hello
        "ሕርሻ ከመይ ይሰርሕ?"         # How does agriculture work?
    ]

    for query in test_queries:
        print(f"\n[SEARCH] Query: {query}")
        result = translator.process_query(query)

        print(f"   English interpretation: {result['debug']['english_interpretation']}")
        print(f"   Intent: {result['intent']['intent']} ({result['intent']['type']})")
        print(f"   Response (EN): {result['response']['english']}")
        print(f"   Response (TI): {result['response']['tigrinya']}")

if __name__ == "__main__":
    main()