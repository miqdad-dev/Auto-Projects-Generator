You are an elite code generator (OpenAI GPT-4o / GPT-4.1). Output a COMPLETE, RUNNABLE “mini-hard” project with non-trivial logic.

OBJECTIVE
- Build a new mini-hard project (beyond trivial examples) from a RANDOM programming field.
- Include GitHub automation to generate and commit NEW projects daily and every 5 hours.

RANDOM PROJECT TYPE (choose 1 per run):
web game; business website; portfolio website; e-commerce site; blog platform; dashboard application; social media app; productivity tool; educational platform; entertainment website; news portal; booking system; chat application; file sharing platform; online calculator.

OUTPUT FORMAT (STRICT)
Return ONLY fenced file blocks, each like:

```path/filename
<file content>
````

WEB PROJECT REQUIREMENTS

1. A root folder: <short-slug> (NO DATES in folder names).
2. **Technology Stack** (choose based on project complexity):
   - **Frontend**: HTML5, CSS3, JavaScript (ES6+)
   - **Backend** (if needed): PHP with MySQL OR Java with Spring Boot
   - **Styling**: Modern CSS frameworks (Bootstrap, Tailwind, or custom responsive design)
   - **Database**: MySQL, PostgreSQL, or SQLite for data storage
   - **APIs**: RESTful endpoints for dynamic functionality

3. **Project Structure**:
```
project-name/
├── index.html                 # Main entry point
├── css/
│   ├── style.css             # Main stylesheet
│   └── responsive.css        # Mobile responsiveness
├── js/
│   ├── main.js              # Core JavaScript functionality
│   ├── utils.js             # Utility functions
│   └── api.js               # API interaction (if applicable)
├── php/ (if backend needed)
│   ├── config.php           # Database configuration
│   ├── api/                 # API endpoints
│   └── includes/            # Shared PHP components
├── assets/
│   ├── images/              # Image resources
│   └── icons/               # Icon files
├── database/
│   └── schema.sql           # Database structure (if needed)
└── README.md                # Comprehensive documentation
```

4. **Core Features Required**:
   - **Responsive Design**: Mobile-first, works on all devices
   - **Interactive Elements**: Forms, buttons, dynamic content
   - **Modern UI/UX**: Clean, professional design with good user experience
   - **Functionality**: Core features that solve a real problem
   - **Data Persistence**: Local storage, session storage, or database integration

AUTOMATION REQUIREMENTS

* Include: `.github/workflows/auto.yml`
* Triggers:

  * `cron: "0 9 * * *"`  # daily at 09:00 UTC
  * `cron: "0 */5 * * *"` # every 5 hours
  * `workflow_dispatch:`
* Workflow steps:

  * checkout
  * setup Python 3.11
  * install deps from `requirements.txt`
  * run `scripts/generate_next.py`
  * configure git user
  * commit & push if changes exist with a Conventional Commit message:
    `feat(auto): add project YYYY-MM-DD-<slug>`
* `scripts/generate_next.py` must:

  * pick a RANDOM field from the list
  * create a NEW dated folder (YYYY-MM-DD-<slug>) with a different project than before
  * call provider API (OpenAI or Anthropic) based on env:

    * `PROVIDER` in {"openai","anthropic"}
    * `OPENAI_API_KEY` for OpenAI (e.g., gpt-4o or gpt-4.1)
    * `ANTHROPIC_API_KEY` for Claude (e.g., claude-3-5-sonnet)
    * optional `MODEL_NAME`
  * embed a robust “file-emitter” prompt that asks the model to output files in the same fenced-filename format, then parse/write them to disk.

WEB PROJECT DESIGN REQUIREMENTS

**Visual Design Variety** (choose different design approach each time):
- **Modern Minimalist**: Clean lines, white space, subtle shadows, monochromatic color schemes
- **Colorful & Vibrant**: Bold colors, gradients, playful animations, energetic feel  
- **Dark Mode**: Dark backgrounds, neon accents, modern tech aesthetic
- **Corporate Professional**: Blue/gray themes, clean layouts, business-focused
- **Creative Portfolio**: Unique layouts, artistic elements, showcase-focused
- **Gaming/Entertainment**: Dynamic backgrounds, interactive elements, engaging visuals
- **E-commerce**: Product-focused, conversion-optimized, trust-building elements

**Technical Complexity Levels**:

**LEVEL 1 - Frontend Only (HTML/CSS/JS):**
- Static websites with interactive JavaScript
- Local storage for data persistence
- APIs integration (weather, news, etc.)
- Complex animations and transitions
- Games using Canvas or WebGL

**LEVEL 2 - Full-Stack PHP:**
- PHP backend with MySQL database
- User authentication and sessions
- CRUD operations and data management
- File uploads and media handling
- RESTful API endpoints

**LEVEL 3 - Java Enterprise:**
- Java Spring Boot backend
- PostgreSQL database with JPA
- MVC architecture pattern
- Advanced security and authentication
- Microservices architecture (when appropriate)

**Problem-Solving Focus** (each project must solve a unique problem):
- **Productivity**: Task management, time tracking, note-taking
- **Entertainment**: Games, quizzes, interactive stories
- **Business**: Inventory management, customer relations, analytics
- **Education**: Learning platforms, quiz systems, progress tracking
- **Social**: Community features, sharing, collaboration
- **Utility**: Calculators, converters, tools, generators
- **Creative**: Art tools, music players, photo editors

**Quality Standards:**
- **Responsive Design**: Mobile-first, tablet and desktop optimized
- **Cross-browser Compatibility**: Works on Chrome, Firefox, Safari, Edge
- **Performance Optimized**: Fast loading, optimized images, minified code
- **Accessibility**: ARIA labels, keyboard navigation, screen reader friendly
- **SEO Optimized**: Meta tags, semantic HTML, structured data
- **Security**: Input validation, XSS protection, secure data handling
- **Code Quality**: Clean, commented, organized code structure

DELIVERABLES

* Provide ALL files (source, README, tests, scripts, workflow, requirements, Makefile/Dockerfile, etc.) ONLY as fenced filename blocks exactly like:

```path/filename
<content>
```

