"""
Task 2: Chatbot for FAQs - Cloud Computing edition
====================================

What this program does (maps 1:1 to the Task 2 steps):
  1. Collect FAQs ........ the FAQS list below (questions + answers + paraphrases)
  2. Preprocess text ..... NLTK (tokenizer + Porter stemmer) with a built-in stop-word list
  3. Match questions ..... TF-IDF vectors + cosine similarity (scikit-learn)
  4. Display best answer . answer + matched question + confidence meter
  5. Optional chat UI .... a Flask web app with a custom chat interface

Run it:
    pip install -r requirements.txt
    python chatbot.py            -> opens the chat UI in your browser
    python chatbot.py --cli      -> chat in the terminal instead
    python chatbot.py --test     -> quick self-test of the matching engine

No NLTK downloads are needed. If NLTK is not installed, the bot falls back to a
small built-in stemmer so it never crashes.
"""

import argparse
import re
import socket
import sys
import threading
import webbrowser

# Make sure printing never crashes on Windows consoles.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --------------------------------------------------------------------------
# Optional / required libraries
# --------------------------------------------------------------------------
try:
    from nltk.stem import PorterStemmer          # needs NO downloaded data
    from nltk.tokenize import RegexpTokenizer    # needs NO downloaded data
    _STEMMER = PorterStemmer()
    _TOKENIZER = RegexpTokenizer(r"[a-z0-9]+")
    NLTK_OK = True
except Exception:                                # nltk missing -> fallback
    _STEMMER = None
    _TOKENIZER = None
    NLTK_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    sys.exit("Missing package. Run:  pip install scikit-learn flask nltk")

BOT_NAME = "Nimbus"
TOPIC = "Cloud Computing"

# --------------------------------------------------------------------------
# STEP 1 - COLLECT FAQs
# Each entry: category, question, answer, alt (other ways people ask it),
# kw (extra keywords). Add your own entries here - nothing else to change.
# --------------------------------------------------------------------------
CATEGORIES = {
    "Cloud basics": "#2F5BFF",
    "Models and services": "#7B4DFF",
    "Providers and tools": "#00997F",
    "Security, cost and careers": "#D98A00",
    "About this chatbot": "#D6336C",
}


def faq(cat, q, a, alt=(), kw=""):
    return {"cat": cat, "q": q, "a": a, "alt": list(alt), "kw": kw}


CB, MS, PT, SC, AB = list(CATEGORIES)

FAQS = [
    # ------------------------- Cloud basics -------------------------
    faq(CB, "What is cloud computing?",
        "Cloud computing means using computing resources such as servers, storage, databases and "
        "software over the internet, instead of owning and running your own hardware. You use what "
        "you need and usually pay only for what you use.",
        ["Define cloud computing", "Explain the cloud", "What does the cloud mean",
         "Introduction to cloud"], "cloud computing definition meaning internet servers"),
    faq(CB, "Why do companies use the cloud?",
        "Companies use the cloud to avoid buying expensive hardware, to scale up or down quickly, to "
        "reach customers in many regions, to get built-in backup and security tools, and to "
        "release products faster.",
        ["Benefits of cloud computing", "Advantages of the cloud", "Why move to the cloud",
         "What are the pros of cloud"], "benefits advantages pros why use"),
    faq(CB, "What are the disadvantages of cloud computing?",
        "Common drawbacks are dependence on internet access, ongoing costs that can grow if not "
        "monitored, less direct control over the underlying hardware, possible vendor lock-in, and "
        "the need to configure security correctly.",
        ["Cons of cloud computing", "Limitations of the cloud", "Risks of cloud",
         "Drawbacks of using cloud"], "disadvantages cons limitations drawbacks risks"),
    faq(CB, "What is virtualization?",
        "Virtualization uses software called a hypervisor to split one physical machine into several "
        "virtual machines, each with its own operating system. It is the technology that makes "
        "sharing cloud hardware between many customers possible.",
        ["Explain virtual machines", "What is a hypervisor", "What is a VM",
         "How do virtual machines work"], "virtualization virtual machine vm hypervisor"),
    faq(CB, "What is a data center?",
        "A data center is a facility full of servers, storage and networking equipment, with power, "
        "cooling and security. Cloud providers run huge data centers around the world and rent out "
        "their capacity.",
        ["What is a datacenter", "Where is my data stored in the cloud", "Cloud server location"],
        "data center datacenter facility servers location"),
    faq(CB, "What is scalability and elasticity in the cloud?",
        "Scalability is the ability to handle more load by adding resources. Elasticity is "
        "scaling up and down automatically as demand changes, for example adding servers during a "
        "sale and removing them afterwards so you stop paying for them.",
        ["Explain scaling", "What is auto scaling", "Difference between scalability and elasticity",
         "How does the cloud handle high traffic"], "scalability elasticity autoscaling scale traffic load"),
    faq(CB, "What are regions and availability zones?",
        "A region is a geographic area where a provider has data centers. Each region has several "
        "availability zones, which are separate data centers. Spreading your app across zones keeps "
        "it running if one zone fails.",
        ["Explain AWS regions", "What is an availability zone", "What is a cloud region",
         "How to get high availability"], "region availability zone az high availability"),
    faq(CB, "What is the difference between cloud and traditional on-premises hosting?",
        "On-premises means you buy, house and maintain your own servers. In the cloud the provider "
        "owns the hardware and you rent it on demand. The cloud is faster to start and easier to "
        "scale, while on-premises gives you full physical control.",
        ["On premise vs cloud", "Cloud versus local servers", "Cloud vs traditional IT",
         "Compare cloud and on-premises"], "on-premises onpremise traditional local hosting compare"),

    # ---------------------- Models and services ----------------------
    faq(MS, "What are IaaS, PaaS and SaaS?",
        "They are the three main cloud service models:\n"
        "- IaaS (Infrastructure as a Service): you rent virtual servers, storage and networks, "
        "e.g. Amazon EC2.\n"
        "- PaaS (Platform as a Service): you deploy your code and the platform manages the servers, "
        "e.g. Google App Engine.\n"
        "- SaaS (Software as a Service): you simply use finished software, e.g. Gmail or Google Docs.",
        ["Explain cloud service models", "Types of cloud services", "What is infrastructure as a service",
         "Difference between IaaS PaaS and SaaS"], "iaas paas saas service models types"),
    faq(MS, "What are public, private and hybrid clouds?",
        "A public cloud is run by a provider and shared by many customers. A private cloud is used "
        "by a single organization. A hybrid cloud combines both so workloads can move between them.",
        ["Cloud deployment models", "Types of cloud deployment", "Explain hybrid cloud",
         "What is a private cloud", "What is a public cloud"], "public private hybrid deployment models community"),
    faq(MS, "What is multi-cloud?",
        "Multi-cloud means using services from more than one cloud provider, for example AWS for "
        "compute and Google Cloud for analytics. It reduces dependence on a single vendor but adds "
        "complexity.",
        ["Explain multicloud", "Using more than one cloud provider", "Multiple clouds"],
        "multi-cloud multicloud multiple providers"),
    faq(MS, "What is serverless computing?",
        "Serverless lets you run code without managing servers. You upload a function, the provider "
        "runs it when triggered and bills you only for the time it runs. AWS Lambda and Azure "
        "Functions are examples.",
        ["Explain serverless", "What is AWS Lambda", "Function as a service", "What is FaaS"],
        "serverless lambda functions faas azure functions"),
    faq(MS, "What are containers and Docker?",
        "A container packages an app with everything it needs to run, so it behaves the same on any "
        "machine. Docker is the most popular tool for building and running containers. Containers "
        "start faster and are lighter than virtual machines.",
        ["Explain Docker", "What is containerization", "Docker vs virtual machine",
         "What is a docker container"], "docker container containerization image"),
    faq(MS, "What is Kubernetes?",
        "Kubernetes is an open-source system that automatically deploys, scales and manages "
        "containers across many machines. If a container crashes, Kubernetes restarts it.",
        ["Explain K8s", "What is container orchestration", "Why use Kubernetes"],
        "kubernetes k8s orchestration cluster pods"),
    faq(MS, "What is cloud storage and what types exist?",
        "Cloud storage keeps your data on a provider's servers. Main types are object storage "
        "(files and media, e.g. Amazon S3), block storage (disks attached to servers, e.g. EBS) "
        "and file storage (shared folders, e.g. Amazon EFS).",
        ["Types of cloud storage", "Object storage vs block storage", "What is Amazon S3",
         "Where can I store files in the cloud"], "storage s3 object block file ebs efs bucket"),
    faq(MS, "What is a CDN?",
        "A Content Delivery Network keeps copies of your website content on servers around the "
        "world, so visitors load it from a location near them. This makes sites faster. "
        "Examples are Amazon CloudFront and Cloudflare.",
        ["Content delivery network", "How do websites load faster worldwide", "What is CloudFront"],
        "cdn content delivery network cloudfront cloudflare"),
    faq(MS, "What is a load balancer?",
        "A load balancer spreads incoming traffic across several servers so that no single server "
        "is overloaded. It also stops sending traffic to servers that are unhealthy.",
        ["Explain load balancing", "How is traffic distributed to servers", "What is ELB"],
        "load balancer balancing traffic distribute elb"),

    # ---------------------- Providers and tools ----------------------
    faq(PT, "Who are the major cloud providers?",
        "The three largest are Amazon Web Services (AWS), Microsoft Azure and Google Cloud "
        "Platform (GCP). Others include IBM Cloud, Oracle Cloud and Alibaba Cloud.",
        ["Top cloud companies", "Which cloud platforms exist", "List of cloud vendors",
         "Biggest cloud providers"], "providers vendors aws azure gcp google microsoft amazon top"),
    faq(PT, "What is AWS?",
        "Amazon Web Services is Amazon's cloud platform. It offers over 200 services including "
        "compute (EC2, Lambda), storage (S3), databases (RDS, DynamoDB) and networking (VPC).",
        ["Explain Amazon Web Services", "Tell me about AWS", "Define AWS"],
        "aws amazon web services"),
    faq(PT, "What is Microsoft Azure?",
        "Azure is Microsoft's cloud platform. It is popular with companies that already use "
        "Windows, Office 365 and Active Directory, and offers virtual machines, storage, "
        "databases, AI services and more.",
        ["Explain Azure", "Tell me about Microsoft Azure", "Define Azure"], "azure microsoft"),
    faq(PT, "What is Google Cloud Platform?",
        "Google Cloud Platform (GCP) is Google's cloud. It is well known for data analytics "
        "(BigQuery), machine learning tools, Kubernetes (which Google created) and global "
        "networking.",
        ["Explain GCP", "Tell me about Google Cloud", "What is BigQuery", "Define GCP"],
        "gcp google cloud platform bigquery"),
    faq(PT, "What is Amazon EC2?",
        "EC2 (Elastic Compute Cloud) lets you rent virtual servers on AWS. You choose the "
        "operating system, CPU, memory and storage, and can start or stop servers in minutes.",
        ["Explain EC2", "What is an EC2 instance", "How to rent a virtual server on AWS"],
        "ec2 instance virtual server compute aws"),
    faq(PT, "Is there a free tier to practise cloud skills?",
        "Yes. AWS, Azure and Google Cloud all offer free tiers or starter credits for new accounts "
        "so you can practise. Free limits and durations change, so read the current terms on the "
        "provider's site and set a billing alert to avoid surprise charges.",
        ["Can I learn cloud for free", "Free AWS account", "Free tier limits",
         "How to practice cloud without paying"], "free tier practice credits trial learn account"),
    faq(PT, "What databases are available in the cloud?",
        "Providers offer managed relational databases (such as Amazon RDS, Azure SQL, Cloud SQL) "
        "and NoSQL databases (such as DynamoDB, Cosmos DB, Firestore). Managed means the provider "
        "handles backups, patching and scaling.",
        ["Cloud database services", "What is RDS", "What is DynamoDB", "Managed database"],
        "database rds dynamodb nosql sql managed"),
    faq(PT, "What is DevOps and CI/CD in the cloud?",
        "DevOps is a way of working where developers and operations teams automate building, "
        "testing and releasing software. CI/CD (continuous integration and delivery) pipelines "
        "automatically test and deploy every code change. Tools include GitHub Actions, Jenkins "
        "and AWS CodePipeline.",
        ["Explain DevOps", "What is a CI CD pipeline", "Automate deployment"],
        "devops cicd pipeline jenkins github actions automate deploy"),
    faq(PT, "What is Infrastructure as Code?",
        "Infrastructure as Code (IaC) means describing your servers and networks in code files "
        "instead of clicking in a console, so they can be created repeatably. Popular tools are "
        "Terraform and AWS CloudFormation.",
        ["Explain Terraform", "What is CloudFormation", "IaC tools"],
        "infrastructure code iac terraform cloudformation"),

    # ------------------ Security, cost and careers -------------------
    faq(SC, "Is cloud computing secure?",
        "Major providers invest heavily in physical and network security, but safety also depends "
        "on how you configure your account. Use strong passwords, multi-factor authentication, "
        "least-privilege access and encryption.",
        ["How safe is the cloud", "Is my data safe in the cloud", "Cloud security",
         "Can cloud data be hacked"], "security secure safe hack encryption protect"),
    faq(SC, "What is the shared responsibility model?",
        "The provider is responsible for the security of the cloud (data centers, hardware, core "
        "services). You are responsible for security in the cloud: your data, user access, "
        "configuration and applications.",
        ["Explain shared responsibility", "Who is responsible for cloud security",
         "Customer vs provider security"], "shared responsibility model security provider customer"),
    faq(SC, "What is IAM?",
        "IAM (Identity and Access Management) controls who can do what in your cloud account. "
        "You create users, groups and roles and give each only the permissions they need, which "
        "is called the principle of least privilege.",
        ["Explain identity and access management", "How to control access in the cloud",
         "What are IAM roles", "Cloud permissions"], "iam identity access management roles permissions users"),
    faq(SC, "How does cloud pricing work?",
        "Most services are pay-as-you-go: you pay for the compute time, storage, and data "
        "transfer you use. Discounts exist for committing to long-term usage (reserved or savings "
        "plans) and for spare capacity (spot instances).",
        ["Pay as you go", "How much does AWS cost", "How much does the cloud cost", "Cloud billing", "What is a spot instance",
         "How are cloud services charged"], "pricing cost billing pay-as-you-go charge spot reserved"),
    faq(SC, "How can I reduce my cloud costs?",
        "Turn off unused resources, choose the right instance size, use auto scaling, buy reserved "
        "or savings plans for steady workloads, store rarely used data in cheaper tiers, and set "
        "budgets and billing alerts.",
        ["Save money on cloud", "How to cut my cloud bill", "Cloud cost optimization", "Why is my cloud bill high",
         "Lower AWS bill"], "reduce save optimize cost bill budget alert cheaper"),
    faq(SC, "What is vendor lock-in?",
        "Vendor lock-in happens when your system depends so heavily on one provider's special "
        "services that moving elsewhere becomes costly. Open standards, containers and "
        "multi-cloud designs help reduce it.",
        ["Explain vendor lock in", "Can I switch cloud providers", "Provider dependency"],
        "vendor lock-in lockin switch migrate dependency"),
    faq(SC, "How do I back up data and recover from disasters in the cloud?",
        "Use automated backups and snapshots, keep copies in more than one region, and plan your "
        "recovery goals: RTO (how fast you must recover) and RPO (how much data loss is "
        "acceptable). Test your recovery plan regularly.",
        ["Cloud backup", "Disaster recovery", "What is RTO and RPO", "Restore lost data"],
        "backup disaster recovery rto rpo snapshot restore"),
    faq(SC, "Which cloud certifications should a beginner get?",
        "Good starting points are AWS Certified Cloud Practitioner, Microsoft Azure Fundamentals "
        "(AZ-900) and Google Cloud Digital Leader. Next steps are associate-level exams such as "
        "AWS Solutions Architect Associate.",
        ["Best cloud certificate for beginners", "Cloud exams", "How to start a cloud career",
         "AWS certification path"], "certification certificate exam beginner az-900 practitioner"),
    faq(SC, "What jobs are there in cloud computing?",
        "Common roles include Cloud Engineer, Cloud Architect, DevOps Engineer, Site Reliability "
        "Engineer, Cloud Security Engineer and Cloud Support Associate. Skills that help: Linux, "
        "networking, Python or scripting, Docker and one major provider.",
        ["Cloud career options", "Careers in cloud", "Skills needed for cloud engineer",
         "How to become a cloud engineer"], "jobs career roles engineer architect skills salary"),
    faq(SC, "How do I migrate an application to the cloud?",
        "Typical approaches are rehost (lift and shift with few changes), replatform (small "
        "optimizations), and refactor (redesign for the cloud). Start by assessing your apps, "
        "pick a pilot project, migrate in stages and test before switching users over.",
        ["Cloud migration steps", "Move to the cloud", "Lift and shift", "Shift my app to AWS"],
        "migration migrate move lift shift rehost refactor"),

    # ----------------------- About this chatbot ----------------------
    faq(AB, "How does this chatbot find the right answer?",
        "1) Your message is cleaned, tokenized, stop words are removed and words are stemmed.\n"
        "2) It is converted to a TF-IDF vector.\n"
        "3) Cosine similarity compares it with every stored FAQ question.\n"
        "4) The FAQ with the highest score is shown. If the score is low, the bot suggests the "
        "closest topics instead of guessing.",
        ["How does the chatbot work", "Explain the working of this bot",
         "What algorithm does the bot use", "How are answers matched"],
        "working algorithm pipeline matching nlp bot"),
    faq(AB, "What is NLTK?",
        "NLTK (Natural Language Toolkit) is a Python library for working with human language. It "
        "provides tokenizers, stemmers, lemmatizers, stop-word lists and many other NLP tools.",
        ["Explain NLTK", "Define NLTK", "What does NLTK do"], "nltk natural language toolkit"),
    faq(AB, "What is cosine similarity?",
        "Cosine similarity measures the angle between two text vectors. A score of 1 means the "
        "texts point in the same direction, and 0 means they share nothing. The bot answers with "
        "the FAQ that has the highest score.",
        ["Explain cosine similarity", "How does the bot compare questions",
         "How is similarity calculated"], "cosine similarity score vector compare"),
    faq(AB, "What is TF-IDF?",
        "TF-IDF turns text into numbers. A word scores high if it appears often in one question but "
        "rarely across all questions, so distinctive words get more weight.",
        ["Explain TF IDF", "What is term frequency", "Define tfidf vectorizer"],
        "tfidf tf-idf vectorizer term frequency inverse document"),
    faq(AB, "What is tokenization and stemming?",
        "Tokenization splits text into words, for example 'How do I scale?' becomes "
        "['how', 'do', 'i', 'scale']. Stemming cuts words to a base form (running becomes run) so "
        "different forms of a word match each other.",
        ["Explain tokenizing", "What is stemming", "What is lemmatization", "What are stop words"],
        "token tokenize stem stemming lemma stopwords split words"),
    faq(AB, "How do I run this chatbot?",
        "1) Install the packages: pip install -r requirements.txt\n"
        "2) Start it: python chatbot.py\n"
        "3) Your browser opens the chat at http://127.0.0.1:5000\n"
        "Prefer the terminal? Use: python chatbot.py --cli",
        ["How to start the chatbot", "Steps to run the project", "Command to run the app",
         "ModuleNotFoundError no module named", "pip install error"],
        "run start execute launch command install pip error"),
    faq(AB, "How do I add my own FAQs to the bot?",
        "Open chatbot.py, find the FAQS list and add a new entry such as:\n"
        "faq(CB, 'Your question?', 'Your answer.', ['another way to ask it'], 'keywords')\n"
        "Save the file and restart the bot. The new FAQ is picked up automatically.",
        ["Add new questions", "Edit the FAQ list", "How to change the dataset",
         "Customize the answers"], "add edit change dataset customize new faq"),
    faq(AB, "Why does the bot not understand my question?",
        "The bot can only match what is in its FAQ list. Try rephrasing with key words, or add "
        "more alternative phrasings for that FAQ in its 'alt' list. You can also lower the "
        "MATCH_THRESHOLD value in chatbot.py.",
        ["The bot gives a wrong answer", "Bot says it does not know", "Low accuracy",
         "Why is the answer incorrect"], "wrong answer accuracy understand fallback"),
    faq(AB, "How can I make the chatbot smarter?",
        "Add more FAQs and alternative phrasings, add synonyms, or switch from TF-IDF to sentence "
        "embeddings (for example the sentence-transformers library) for better meaning-based "
        "matching.",
        ["Improve accuracy", "Make the bot better", "Upgrade the chatbot", "Advanced features"],
        "improve smarter better upgrade embeddings bert"),
]

# Small talk handled before FAQ matching.
SMALLTALK = [
    (r"^(hi+|hello+|hey+|hii+|yo|namaste|good (morning|afternoon|evening))\b",
     "Hello! I'm Nimbus, your cloud computing help desk. Ask me anything or tap a topic on the left."),
    (r"\b(thanks|thank you|thx|ty|thank u)\b",
     "You're welcome! Ask me anything else whenever you like."),
    (r"\b(bye|goodbye|see you|see ya|exit|quit)\b",
     "Goodbye, and happy learning!"),
    (r"(who are you|your name|what can you do|what are you)",
     "I'm Nimbus, an FAQ chatbot about cloud computing. I compare your question with a list of "
     "stored FAQs using NLP (NLTK + TF-IDF + cosine similarity) and show the best match."),
]

# --------------------------------------------------------------------------
# STEP 2 - PREPROCESS TEXT
# --------------------------------------------------------------------------
STOP_WORDS = set("""
a about above after again all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its itself just me more most my myself no nor of
off on once only or other our ours out over own same she should so some such than that the their
theirs them then there these they this those through to too under until up us very was we were
what when where which while who whom why will with would you your yours yourself please tell
want need get got let know like also
""".split())

SYNONYMS = {
    "cert": "certification", "certificate": "certification", "certificates": "certification",
    "certifications": "certification", "exam": "certification", "exams": "certification",
    "pricing": "cost", "price": "cost", "bill": "cost", "billing": "cost", "charge": "cost",
    "charged": "cost", "cheap": "cost", "cheaper": "cost", "expensive": "cost",
    "safe": "security", "secure": "security", "safety": "security", "hack": "security",
    "hacked": "security",
    "k8s": "kubernetes", "vm": "virtual machine", "vms": "virtual machine",
    "datacenter": "data center", "multicloud": "multi cloud",
    "bot": "chatbot", "chat": "chatbot",
    "mistake": "error", "bug": "error",
    "libs": "library", "libraries": "library", "package": "library", "packages": "library",
    "tfidf": "tf idf", "stopwords": "stop words", "stopword": "stop words",
    "job": "career", "jobs": "career",
}


def _fallback_stem(word):
    """Tiny stemmer used only when NLTK is not installed."""
    for suffix in ("ingly", "edly", "ing", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokenize(text):
    if NLTK_OK:
        return _TOKENIZER.tokenize(text)
    return re.findall(r"[a-z0-9]+", text)


def _stem(word):
    if NLTK_OK:
        try:
            return _STEMMER.stem(word)
        except Exception:
            return _fallback_stem(word)
    return _fallback_stem(word)


def preprocess(text):
    """lowercase -> remove urls -> tokenize -> synonyms -> stop words -> stem."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("'", "").replace("\u2019", "")
    tokens = []
    for tok in _tokenize(text):
        tok = SYNONYMS.get(tok, tok)
        tokens.extend(tok.split())
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(_stem(t) for t in tokens)


# --------------------------------------------------------------------------
# STEP 3 + 4 - MATCH USING COSINE SIMILARITY AND RETURN THE BEST ANSWER
# --------------------------------------------------------------------------
MATCH_THRESHOLD = 0.30     # at or above: confident answer
SUGGEST_THRESHOLD = 0.12   # between: "did you mean...?"


class FAQBot:
    def __init__(self, faqs):
        self.faqs = faqs
        rows, owner = [], []
        for i, item in enumerate(faqs):
            variants = [item["q"]] + item["alt"]
            if item["kw"]:
                variants.append(item["kw"])
            variants.append(item["q"] + " " + item["kw"])
            for v in variants:
                clean = preprocess(v)
                if clean:
                    rows.append(clean)
                    owner.append(i)
        self.owner = owner
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                          token_pattern=r"(?u)\b\w+\b")
        self.matrix = self.vectorizer.fit_transform(rows)

    def _scores(self, text):
        clean = preprocess(text)
        if not clean:
            return None
        sims = cosine_similarity(self.vectorizer.transform([clean]), self.matrix)[0]
        best = [0.0] * len(self.faqs)
        for sim, idx in zip(sims, self.owner):
            if sim > best[idx]:
                best[idx] = float(sim)
        return best

    def starters(self, n=4):
        wanted = ["What is cloud computing?", "What are IaaS, PaaS and SaaS?",
                  "Who are the major cloud providers?", "Is cloud computing secure?"]
        qs = [q for q in wanted if any(f["q"] == q for f in self.faqs)]
        return qs[:n]

    def reply(self, message):
        message = (message or "").strip()
        if not message:
            return {"kind": "fallback", "answer": "Type a question and I'll look for the best match.",
                    "related": self.starters()}
        if len(message) > 500:
            message = message[:500]

        # small talk first (short messages only)
        if len(message.split()) <= 6:
            low = message.lower()
            for pattern, answer in SMALLTALK:
                if re.search(pattern, low):
                    return {"kind": "smalltalk", "answer": answer, "related": self.starters()}

        scores = self._scores(message)
        if scores is None:
            return {"kind": "fallback",
                    "answer": "I couldn't find any keywords in that. Try asking about cloud models, "
                              "providers, security, costs or this chatbot.",
                    "related": self.starters()}

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = ranked[0]
        top_score = scores[top]
        others = [self.faqs[i]["q"] for i in ranked[1:4] if scores[i] >= 0.08]

        if top_score >= MATCH_THRESHOLD:
            item = self.faqs[top]
            if not others:
                others = [f["q"] for f in self.faqs if f["cat"] == item["cat"] and f["q"] != item["q"]][:3]
            return {"kind": "answer", "answer": item["a"], "matched": item["q"],
                    "category": item["cat"], "color": CATEGORIES[item["cat"]],
                    "confidence": round(min(top_score, 1.0) * 100), "related": others}

        if top_score >= SUGGEST_THRESHOLD:
            picks = [self.faqs[i]["q"] for i in ranked[:3] if scores[i] >= SUGGEST_THRESHOLD]
            return {"kind": "suggest",
                    "answer": "I'm not fully sure what you mean. Did you mean one of these?",
                    "related": picks}

        return {"kind": "fallback",
                "answer": "Sorry, I don't have an answer for that yet. Try different keywords, or "
                          "pick one of these popular questions.",
                "related": self.starters()}

    def topics(self):
        out = []
        for name, color in CATEGORIES.items():
            out.append({"name": name, "color": color,
                        "questions": [f["q"] for f in self.faqs if f["cat"] == name]})
        return out


# --------------------------------------------------------------------------
# STEP 5 (OPTIONAL) - CHAT UI
# --------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nimbus - Cloud Computing FAQ Chatbot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#EEF1F8; --grid:#DCE2F0; --ink:#13203F; --ink-soft:#4A5878; --card:#FFFFFF;
  --blue:#2F5BFF; --sun:#FFD23F; --line:#C9D1E6; --danger:#C0392B;
  --display:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif;
  --body:"IBM Plex Sans","Segoe UI",system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  font-family:var(--body);color:var(--ink);background-color:var(--paper);
  background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:28px 28px;display:flex;justify-content:center;padding:24px;
}
.shell{width:100%;max-width:1120px;height:calc(100vh - 48px);min-height:520px;display:grid;
  grid-template-columns:320px 1fr;background:var(--card);border:2px solid var(--ink);
  border-radius:18px;box-shadow:10px 10px 0 var(--ink);overflow:hidden}
/* ---------- side panel ---------- */
.side{background:var(--ink);color:#E8ECF8;display:flex;flex-direction:column;min-height:0}
.brand{padding:22px 22px 16px}
.brand h1{font-family:var(--display);font-weight:800;font-size:30px;line-height:1;letter-spacing:-.5px}
.brand h1 span{background:var(--sun);color:var(--ink);padding:0 8px;border-radius:6px;margin-left:2px}
.brand p{margin-top:10px;font-size:14px;color:#AEB9D8;line-height:1.45}
.topics{flex:1;overflow-y:auto;padding:4px 14px 16px}
details{border-top:1px solid #2C3C66}
details:last-child{border-bottom:1px solid #2C3C66}
summary{list-style:none;cursor:pointer;padding:13px 8px;display:flex;align-items:center;gap:10px;
  font-family:var(--display);font-weight:700;font-size:16px}
summary::-webkit-details-marker{display:none}
summary .dot{width:12px;height:12px;border-radius:3px;flex:none}
summary .count{margin-left:auto;font-family:var(--body);font-weight:500;font-size:12px;color:#8E9BC0}
summary:focus-visible,.topics button:focus-visible,.chip:focus-visible,.send:focus-visible,.clear:focus-visible,textarea:focus-visible{outline:3px solid var(--sun);outline-offset:2px}
details[open] summary{color:#fff}
.topics button{display:block;width:100%;text-align:left;background:none;border:0;color:#C7D0EA;
  font:inherit;font-size:14px;line-height:1.35;padding:8px 8px 8px 30px;border-radius:8px;cursor:pointer}
.topics button:hover{background:#22325C;color:#fff}
/* ---------- chat ---------- */
.chat{display:flex;flex-direction:column;min-height:0;min-width:0}
.top{display:flex;align-items:center;gap:12px;padding:14px 22px;border-bottom:2px solid var(--ink);background:#F7F9FF}
.avatar{width:40px;height:40px;border-radius:12px;background:var(--blue);color:#fff;display:grid;place-items:center;
  font-family:var(--display);font-weight:800;font-size:22px;border:2px solid var(--ink)}
.who strong{font-family:var(--display);font-size:18px;display:block;line-height:1.1}
.who small{font-size:13px;color:var(--ink-soft)}
.who small::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:#16A870;margin-right:6px}
.clear{margin-left:auto;background:none;border:2px solid var(--ink);color:var(--ink);font:inherit;font-weight:600;
  font-size:13px;padding:7px 12px;border-radius:10px;cursor:pointer}
.clear:hover{background:var(--sun)}
.log{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:18px;scroll-behavior:smooth}
.row{display:flex;flex-direction:column;max-width:82%}
.row.user{align-self:flex-end;align-items:flex-end}
.bubble{padding:12px 16px;font-size:15.5px;line-height:1.55;white-space:pre-line;overflow-wrap:anywhere}
.user .bubble{background:var(--ink);color:#fff;border-radius:16px 16px 4px 16px}
.bot .bubble{background:#fff;border:2px solid var(--ink);border-left:8px solid var(--accent,var(--blue));border-radius:4px 16px 16px 16px}
.matched{margin-top:8px;font-size:13px;color:var(--ink-soft);display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center}
.matched b{color:var(--ink);font-weight:600}
.meter{display:inline-flex;align-items:center;gap:8px}
.meter i{display:block;width:90px;height:8px;border-radius:99px;background:var(--grid);overflow:hidden;border:1px solid var(--line)}
.meter i span{display:block;height:100%;background:var(--accent,var(--blue));border-radius:99px;transition:width .5s ease}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.chip{font:inherit;font-size:13.5px;background:#fff;color:var(--ink);border:2px solid var(--ink);border-radius:99px;padding:6px 13px;cursor:pointer;text-align:left}
.chip:hover{background:var(--sun)}
.hint{font-size:12.5px;color:var(--ink-soft);margin-top:10px}
.typing{display:flex;gap:5px;padding:14px 16px;background:#fff;border:2px solid var(--ink);border-left:8px solid var(--blue);border-radius:4px 16px 16px 16px;width:max-content}
.typing span{width:8px;height:8px;border-radius:50%;background:var(--ink);animation:bounce 1s infinite}
.typing span:nth-child(2){animation-delay:.15s}.typing span:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,60%,100%{transform:translateY(0);opacity:.35}30%{transform:translateY(-6px);opacity:1}}
.composer{display:flex;gap:10px;padding:14px 18px;border-top:2px solid var(--ink);background:#F7F9FF}
textarea{flex:1;resize:none;font:inherit;font-size:15.5px;padding:11px 14px;border:2px solid var(--ink);border-radius:12px;
  background:#fff;color:var(--ink);height:48px;max-height:120px}
.send{background:var(--blue);color:#fff;border:2px solid var(--ink);border-radius:12px;font:inherit;font-weight:700;
  font-size:15px;padding:0 22px;cursor:pointer;box-shadow:3px 3px 0 var(--ink)}
.send:hover{background:#1F45D6}
.send:active{transform:translate(2px,2px);box-shadow:1px 1px 0 var(--ink)}
.send:disabled{opacity:.6;cursor:not-allowed}
.bot.fresh .bubble{animation:pop .35s ease}
@keyframes pop{from{transform:scale(.97);opacity:0}to{transform:none;opacity:1}}
@media (max-width:820px){
  body{padding:0}
  .shell{height:100vh;border-radius:0;border:0;box-shadow:none;grid-template-columns:1fr;grid-template-rows:auto 1fr}
  .side{max-height:38vh}
  .row{max-width:94%}
  .brand{padding:14px 16px 8px}.brand p{display:none}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style>
</head>
<body>
<main class="shell">
  <aside class="side" aria-label="FAQ topics">
    <div class="brand">
      <h1>Cloud<span>Desk</span></h1>
      <p>Cloud computing help desk. Pick any question below or type your own.</p>
    </div>
    <nav class="topics" id="topics"></nav>
  </aside>

  <section class="chat" aria-label="Chat">
    <header class="top">
      <div class="avatar" aria-hidden="true">N</div>
      <div class="who"><strong>Nimbus</strong><small>FAQ chatbot, online</small></div>
      <button class="clear" id="clear" type="button">New chat</button>
    </header>
    <div class="log" id="log" aria-live="polite"></div>
    <form class="composer" id="form" autocomplete="off">
      <textarea id="input" rows="1" maxlength="500" placeholder="Ask about AWS, Docker, security, costs..." aria-label="Your question"></textarea>
      <button class="send" id="send" type="submit">Send</button>
    </form>
  </section>
</main>

<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let busy = false;

function el(tag, cls, text){
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function scrollDown(){ log.scrollTop = log.scrollHeight; }
const delay = ms => new Promise(r => setTimeout(r, ms));

function addUser(text){
  const row = el('div', 'row user');
  row.appendChild(el('div', 'bubble', text));
  log.appendChild(row); scrollDown();
}

function chipList(questions){
  const wrap = el('div', 'chips');
  questions.forEach(q => {
    const b = el('button', 'chip', q);
    b.type = 'button';
    b.addEventListener('click', () => ask(q));
    wrap.appendChild(b);
  });
  return wrap;
}

function addBot(d){
  const row = el('div', 'row bot fresh');
  if (d.color) row.style.setProperty('--accent', d.color);
  row.appendChild(el('div', 'bubble', d.answer));
  if (d.kind === 'answer'){
    const m = el('div', 'matched');
    const q = el('span'); q.appendChild(document.createTextNode('Matched FAQ: ')); q.appendChild(el('b', '', d.matched));
    const meter = el('span', 'meter');
    const bar = el('i'); const fill = el('span'); fill.style.width = '0%'; bar.appendChild(fill);
    meter.appendChild(document.createTextNode('Confidence ' + d.confidence + '%'));
    meter.appendChild(bar);
    m.appendChild(q); m.appendChild(meter);
    row.appendChild(m);
    setTimeout(() => { fill.style.width = d.confidence + '%'; }, 60);
  }
  if (d.related && d.related.length){
    const label = d.kind === 'answer' ? 'You may also want to ask:' : (d.kind === 'suggest' ? 'Closest matches:' : 'Try one of these:');
    row.appendChild(el('div', 'hint', label));
    row.appendChild(chipList(d.related));
  }
  log.appendChild(row); scrollDown();
}

function showTyping(){
  const row = el('div', 'row bot'); row.id = 'typing';
  const t = el('div', 'typing'); t.appendChild(el('span')); t.appendChild(el('span')); t.appendChild(el('span'));
  row.appendChild(t); log.appendChild(row); scrollDown();
}
function hideTyping(){ const t = document.getElementById('typing'); if (t) t.remove(); }

async function ask(text){
  text = (text || '').trim();
  if (!text || busy) return;
  busy = true; sendBtn.disabled = true;
  addUser(text); input.value = ''; input.style.height = '48px';
  showTyping();
  try {
    const res = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    if (!res.ok) throw new Error('Server error ' + res.status);
    const data = await res.json();
    await delay(450);
    hideTyping(); addBot(data);
  } catch (err) {
    hideTyping();
    addBot({kind: 'error', answer: 'I could not reach the server. Make sure chatbot.py is still running in your terminal, then try again.'});
  }
  busy = false; sendBtn.disabled = false; input.focus();
}

function welcome(){
  addBot({kind: 'smalltalk', answer: "Hi, I'm Nimbus. Ask me about cloud computing: service models, AWS and Azure, security, costs, careers, or how this chatbot works.",
    related: ['What is cloud computing?', 'What are IaaS, PaaS and SaaS?', 'What is serverless computing?', 'How does this chatbot find the right answer?']});
}

form.addEventListener('submit', e => { e.preventDefault(); ask(input.value); });
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); ask(input.value); }
});
input.addEventListener('input', () => {
  input.style.height = '48px';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
});
document.getElementById('clear').addEventListener('click', () => { log.innerHTML = ''; welcome(); });

fetch('/api/topics').then(r => r.json()).then(topics => {
  const nav = document.getElementById('topics');
  topics.forEach((t, i) => {
    const d = document.createElement('details');
    if (i === 0) d.open = true;
    const s = document.createElement('summary');
    const dot = el('span', 'dot'); dot.style.background = t.color;
    s.appendChild(dot); s.appendChild(document.createTextNode(t.name));
    s.appendChild(el('span', 'count', String(t.questions.length)));
    d.appendChild(s);
    t.questions.forEach(q => {
      const b = el('button', '', q); b.type = 'button';
      b.addEventListener('click', () => ask(q));
      d.appendChild(b);
    });
    nav.appendChild(d);
  });
}).catch(() => {});

welcome();
input.focus();
</script>
</body>
</html>
"""


def free_port(preferred):
    """Return preferred port if free, otherwise the next free one."""
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


def create_app(bot):
    from flask import Flask, jsonify, request, Response

    app = Flask(__name__)

    @app.get("/")
    def home():
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/topics")
    def topics():
        return jsonify(bot.topics())

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        return jsonify(bot.reply(data.get("message", "")))

    return app


def run_cli(bot):
    print(f"\n{BOT_NAME}: Hi! Ask me about the {TOPIC}. Type 'exit' to quit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{BOT_NAME}: Goodbye!")
            return
        if text.lower() in {"exit", "quit", "q"}:
            print(f"{BOT_NAME}: Goodbye!")
            return
        r = bot.reply(text)
        print(f"\n{BOT_NAME}: {r['answer']}")
        if r["kind"] == "answer":
            print(f"   [matched: {r['matched']} | confidence: {r['confidence']}%]")
        if r.get("related"):
            print("   Related:")
            for q in r["related"]:
                print(f"    - {q}")
        print()


def run_tests(bot):
    cases = [
        ("explain iaas paas saas", "What are IaaS, PaaS and SaaS?"),
        ("what is cosine similarity", "What is cosine similarity?"),
        ("is my data safe on the cloud", "Is cloud computing secure?"),
        ("how much does aws cost", "How does cloud pricing work?"),
        ("which certificate should a beginner take", "Which cloud certifications should a beginner get?"),
        ("what is lambda", "What is serverless computing?"),
        ("docker vs virtual machine", "What are containers and Docker?"),
        ("how to cut my cloud bill", "How can I reduce my cloud costs?"),
        ("who are the top cloud vendors", "Who are the major cloud providers?"),
        ("what is k8s", "What is Kubernetes?"),
        ("benefits of the cloud", "Why do companies use the cloud?"),
        ("how to become a cloud engineer", "What jobs are there in cloud computing?"),
    ]
    ok = 0
    for text, expected in cases:
        r = bot.reply(text)
        got = r.get("matched")
        good = got == expected
        ok += good
        print(("PASS " if good else "FAIL ") + f"{text!r} -> {got} ({r.get('confidence')}%)")
    print(f"\n{ok}/{len(cases)} passed. NLTK in use: {NLTK_OK}")
    # every FAQ must find itself
    miss = [f["q"] for f in bot.faqs if bot.reply(f["q"]).get("matched") != f["q"]]
    print("FAQs that do not match themselves:", miss or "none")
    print("Small talk:", bot.reply("hello")["kind"], "| Gibberish:", bot.reply("zzzz qqqq")["kind"])


def main():
    parser = argparse.ArgumentParser(description="Cloud Computing FAQ Chatbot")
    parser.add_argument("--cli", action="store_true", help="chat in the terminal")
    parser.add_argument("--test", action="store_true", help="run a quick self-test")
    parser.add_argument("--port", type=int, default=5000, help="preferred port (default 5000)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    bot = FAQBot(FAQS)

    if args.test:
        run_tests(bot)
        return
    if args.cli:
        run_cli(bot)
        return

    try:
        app = create_app(bot)
    except ImportError:
        print("Flask is not installed, starting terminal mode instead.")
        print("For the web UI run:  pip install flask\n")
        run_cli(bot)
        return

    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}"
    print(f"\n{BOT_NAME} is running ({len(FAQS)} FAQs loaded, NLTK: {'yes' if NLTK_OK else 'fallback'}).")
    print(f"Open this address in your browser: {url}")
    print("Press CTRL+C to stop.\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
