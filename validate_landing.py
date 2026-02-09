import requests
from bs4 import BeautifulSoup
import sys
import json

def analyze_site(url):
    print(f"🕵️ ANALIZANDO ADN DIGITAL: {url}")
    try:
        if not url.startswith('http'):
            url = 'https://' + url
            
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
        soup = BeautifulSoup(response.content, 'html.parser')
        
        text = soup.get_text().lower()
        title = soup.title.string.lower() if soup.title else ""
        
        # KEYWORDS DE PERFIL PERSONAL
        personal_indicators = [
            "about me", "sobre mí", "my work", "mi trabajo", "cv", "resume", "curriculum",
            "portfolio", "portafolio", "projects", "proyectos", "skills", "habilidades",
            "contact me", "contactame", "i am", "soy", "experience", "experiencia"
        ]
        
        # KEYWORDS DE BLOQUEO (CORPORATIVO / OTRO)
        block_indicators = [
            "shopping cart", "añadir al carrito", "pricing", "precios", "corporate", "investors",
            "our team", "nuestro equipo", "press release", "news", "noticias", "store", "tienda"
        ]
        
        score = 0
        found_personal = []
        for word in personal_indicators:
            if word in text or word in title:
                score += 1
                found_personal.append(word)
                
        is_blocked = False
        found_blocks = []
        for word in block_indicators:
            if word in text:
                is_blocked = True
                found_blocks.append(word)
                
        # CRITERIO DE VALIDACIÓN
        is_valid = score >= 3 and not is_blocked
        
        result = {
            "url": url,
            "is_personal_landing": is_valid,
            "score": score,
            "indicators": found_personal,
            "blocked_by": found_blocks if is_blocked else None,
            "verdict": "VALS: LANDING PERSONAL DETECTADA" if is_valid else "ERROR: WEB NO PERSONAL O CORPORATIVA"
        }
        
        print(f"VERDICTO: {result['verdict']}")
        return result
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
        res = analyze_site(url)
        print(json.dumps(res, indent=4))
    else:
        print("Uso: python3 validate_landing.py <url>")
