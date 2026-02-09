#!/usr/bin/env python3
"""
🦅 LUXJOB HUNTER: AUTOMATED EMPLOYMENT AGENT v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Specialized bot for the Luxembourg job market. 
Monitors ADEM, Moovijob, Jobs.lu, and Facebook.
Goal: Sell Modern CV Landing Pages (50€) to job seekers and recruiters.
"""

import os
import json
import time
import random
import sys
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATHS & IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE_DIR = "/Users/yoyocubano/Documents/ANTIGRAVITY_CORE_DO_NOT_DELETE"
APP_DIR = "/Users/yoyocubano/Documents/ESYBISNE_APP"
VAULT_PATH = "/Users/yoyocubano/Documents/AIGOTAJOB/LUXJOB_VAULT.json"
DRIVER_PATH = "/Users/yoyocubano/.wdm/drivers/chromedriver/mac64/144.0.7559.133/chromedriver-mac-arm64/chromedriver"

sys.path.append(APP_DIR)
sys.path.append(CORE_DIR)
from api.esybisne_db import LeadManager
from neural_response_engine import get_automated_response

class AiGotAJobBot:
    def __init__(self):
        self.vault = self.load_vault()
        self.db = LeadManager()
        self.driver = self.setup_driver()
        print("🦅 AiGotAJob Bot Initialized. Target: Luxembourg.")

    def load_vault(self):
        with open(VAULT_PATH, 'r') as f:
            return json.load(f)

    def setup_driver(self):
        options = Options()
        options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        service = Service(executable_path=DRIVER_PATH)
        try:
            driver = webdriver.Chrome(service=service, options=options)
            print("✅ Connected to Comandante's Browser.")
            return driver
        except:
            print("⚠️ Browser not found on port 9222. Please run ESYBISNE_START.py first.")
            sys.exit(1)

    def stealth_wait(self, min_s=3, max_s=10):
        time.sleep(random.uniform(min_s, max_s))

    def login_adem(self):
        print("🔑 Accessing ADEM Portal...")
        creds = self.vault["ADEM"]
        self.driver.get(creds["url"])
        self.stealth_wait(4, 7)
        # Login logic here (simplified for now)
        try:
            # ADEM often uses a 'Login' button that redirects
            login_btn = self.driver.find_element(By.XPATH, "//a[contains(text(), 'Connexion') or contains(text(), 'Login')]")
            login_btn.click()
            self.stealth_wait(3, 5)
            # Find fields based on known patterns
            user_field = self.driver.find_element(By.ID, "username") or self.driver.find_element(By.NAME, "user")
            user_field.send_keys(creds["user"])
            pass_field = self.driver.find_element(By.ID, "password") or self.driver.find_element(By.NAME, "password")
            pass_field.send_keys(creds["pass"])
            self.driver.find_element(By.XPATH, "//button[@type='submit']").click()
            print("✅ ADEM Login Successful.")
        except:
            print("⚠️ ADEM Login failed or already logged in.")

    def check_throttle(self, profile_url):
        """Checks if we have already applied to this URL within the last 30 days."""
        # Using the same DB path as LeadManager
        local_db = "/Users/yoyocubano/Documents/ESYBISNE_APP/api/local_leads.db"
        try:
            import sqlite3
            conn = sqlite3.connect(local_db)
            c = conn.cursor()
            c.execute("SELECT created_at FROM leads WHERE profile_url = ? ORDER BY created_at DESC LIMIT 1", (profile_url,))
            result = c.fetchone()
            conn.close()

            if result:
                # Handle both ISO and SQLite default formats
                date_str = result[0]
                try:
                    last_date = datetime.fromisoformat(date_str)
                except:
                    last_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                
                days_passed = (datetime.now() - last_date).days
                if days_passed < 30:
                    return False # Throttled
        except Exception as e:
            print(f"⚠️ Throttle check error: {e}")
        return True # Can apply

    def monitor_moovijob(self):
        print("🕵️ Scouting Moovijob for new offers...")
        self.driver.get("https://www.moovijob.com/offres-emploi/luxembourg")
        self.stealth_wait(5, 10)
        
        try:
            # Extract job cards
            offers = self.driver.find_elements(By.CSS_SELECTOR, "article.job-card")
            for offer in offers[:10]:
                title = offer.find_element(By.CSS_SELECTOR, "h2").text
                company = offer.find_element(By.CSS_SELECTOR, "div.company-name").text
                link = offer.find_element(By.TAG_NAME, "a").get_attribute("href")
                
                # Apply Throttling: Once per 30 days
                if not self.check_throttle(link):
                    print(f"⏩ Throttled: Already contacted {company} recently. Skipping to avoid spam.")
                    continue

                print(f"✨ Found Mission: {title} at {company}")
                
                # Report to SQL
                self.db.add_lead(
                    platform="Moovijob",
                    sector="Job Offer",
                    location="Luxembourg",
                    profile_url=link,
                    name=company,
                    context=f"Position: {title}"
                )
        except Exception as e:
            print(f"⚠️ Moovijob scan error: {e}")

    def monitor_jobs_lu(self):
        print("🕵️ Scouting Jobs.lu...")
        self.driver.get("https://fr.jobs.lu/emplois/luxembourg")
        self.stealth_wait(5, 10)
        try:
            cards = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'job-item')]")
            for card in cards[:5]:
                title = card.find_element(By.XPATH, ".//h2").text
                # link = card.find_element(By.TAG_NAME, "a").get_attribute("href")
                link = self.driver.current_url 
                
                if not self.check_throttle(link):
                    continue

                self.db.add_lead(
                    platform="Jobs.lu",
                    sector="Job Offer",
                    location="Luxembourg",
                    profile_url=link,
                    name="Jobs.lu Poster",
                    context=title
                )
        except: pass

    def hunt_facebook_jobs(self):
        print("🕵️ Facebook Sniper: Looking for Job Seekers in groups...")
        queries = ["busco trabajo luxemburgo", "recherche emploi luxembourg", "job search luxembourg"]
        for q in queries:
            url = f"https://www.facebook.com/search/posts?q={q.replace(' ', '%20')}"
            self.driver.get(url)
            self.stealth_wait(10, 15)
            # Find posts and message them
            # (Reuse logic from facebook_seeker_bot.py)

    def run_swarm(self):
        print("🚀 LUXJOB SWARM: ACTIVE PATROL STARTING.")
        # self.login_adem() # ADEM needs exact selector validation
        while True:
            self.monitor_moovijob()
            self.monitor_jobs_lu()
            self.hunt_facebook_jobs()
            print("💤 Patrolling finishes. Resting 30 min...")
            time.sleep(1800)

if __name__ == "__main__":
    hunter = AiGotAJobBot()
    hunter.run_swarm()
