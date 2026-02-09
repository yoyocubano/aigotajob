/**
 * BEE ARMY I18N MODULE
 * Inherited from Welux Events Architecture
 * Supports: en, es, fr, de, lb, pt
 */

const AVAILABLE_LANGS = ['en', 'es', 'fr', 'de', 'lb', 'pt'];
const DEFAULT_LANG = 'en';

class BeeArmyI18n {
    constructor() {
        this.currentLang = this.detectLanguage();
        this.translations = {};
    }

    detectLanguage() {
        const browserLang = navigator.language.split('-')[0];
        return AVAILABLE_LANGS.includes(browserLang) ? browserLang : DEFAULT_LANG;
    }

    async loadTranslations(lang) {
        try {
            const response = await fetch(`locales/${lang}/translation.json`);
            if (!response.ok) throw new Error(`Failed to load ${lang}`);
            this.translations = await response.json();
            this.currentLang = lang;
            this.applyTranslations();
            document.documentElement.lang = lang;
        } catch (e) {
            console.error("I18N Error:", e);
            if (lang !== DEFAULT_LANG) this.loadTranslations(DEFAULT_LANG);
        }
    }

    applyTranslations() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            const key = el.getAttribute('data-i18n');
            if (this.translations[key]) {
                if (el.tagName === 'INPUT' && el.getAttribute('placeholder')) {
                    el.placeholder = this.translations[key];
                } else if (el.tagName === 'IMG' && el.getAttribute('alt')) {
                    el.alt = this.translations[key];
                } else {
                    el.innerText = this.translations[key];
                }
            }
        });
        
        // Update Title
        if (this.translations['app_title']) {
            document.title = this.translations['app_title'];
        }
    }
    
    init() {
        this.loadTranslations(this.currentLang);
    }
}

// Auto-init
window.BeeI18n = new BeeArmyI18n();
document.addEventListener('DOMContentLoaded', () => window.BeeI18n.init());
