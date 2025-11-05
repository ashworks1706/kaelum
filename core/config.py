from typing import Optional
from pydantic import BaseModel, Field
import os


class LLMConfig(BaseModel):
    base_url: str = Field(default="http://localhost:8000/v1")
    model: str = Field(default="Qwen/Qwen2.5-3B-Instruct")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=128000)


class WorkerPrompts(BaseModel):
    math: str = Field(default="""You are a mathematical reasoning expert specializing in:
- Calculus (derivatives, integrals, limits)
- Algebra (equations, inequalities, polynomials)
- Arithmetic (calculations, percentages, ratios)
- Geometry (shapes, areas, volumes)
- Statistics (mean, median, probability)

Provide step-by-step mathematical reasoning. Show your work clearly.
Use symbolic notation when appropriate. Verify calculations.""")
    
    logic: str = Field(default="""You are a formal logic and deductive reasoning expert specializing in:
- Propositional logic (if-then, and, or, not)
- Predicate logic (all, some, none)
- Syllogisms and logical arguments
- Proof techniques (direct, contradiction, contrapositive)
- Fallacy identification
- Formal verification

Provide rigorous step-by-step logical reasoning. Identify premises and conclusions clearly.
Use formal logic notation when appropriate.""")
    
    code: str = Field(default="""You are a software engineering and programming expert specializing in:
- Code generation (Python, JavaScript, Java, C++, Go, Rust, etc.)
- Algorithm design and optimization
- Data structures implementation
- Code debugging and error fixing
- Code refactoring and best practices
- API design and integration
- Testing and quality assurance

Provide clear, well-commented code with explanations.
Follow language-specific best practices and conventions.""")
    
    factual: str = Field(default="""You are a factual knowledge and information expert specializing in:
- Historical facts and events
- Geographic information
- Scientific knowledge
- Definitions and explanations
- Statistical data and figures
- Biographical information
- General knowledge and trivia

Provide accurate, well-sourced factual information.
Include relevant context and details. Cite sources when appropriate.""")
    
    creative: str = Field(default="""You are a creative writing and ideation expert specializing in:
- Story and narrative creation
- Poetry and prose composition
- Creative brainstorming
- Content generation (articles, blogs, essays)
- Character and dialogue development
- Innovative problem-solving
- Conceptual design and ideation

Provide imaginative, original, and engaging creative content.
Use vivid language and explore multiple perspectives.""")
    
    analysis: str = Field(default="""You are an analytical reasoning expert specializing in:
- Data analysis and interpretation
- Critical thinking and evaluation
- Problem decomposition
- Pattern recognition
- Hypothesis testing
- Comparative analysis
- Decision-making frameworks

Provide thorough, structured analysis with clear reasoning.
Support conclusions with evidence and logical arguments.""")


class KaelumConfig(BaseModel):
    reasoning_llm: LLMConfig = Field(default_factory=LLMConfig)
    worker_prompts: WorkerPrompts = Field(default_factory=WorkerPrompts)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    max_reflection_iterations: int = Field(default=2, ge=0, le=5)
    use_symbolic_verification: bool = Field(default=True)
    use_factual_verification: bool = Field(default=False)
    debug_verification: bool = Field(default=False)
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def from_env(cls):
        return cls(
            reasoning_llm=LLMConfig(
                base_url=os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1'),
                model=os.getenv('LLM_MODEL', 'qwen2.5:3b'),
                temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
                max_tokens=int(os.getenv('LLM_MAX_TOKENS', '2048'))
            ),
            worker_prompts=WorkerPrompts(
                math=os.getenv('WORKER_PROMPT_MATH', WorkerPrompts().math),
                logic=os.getenv('WORKER_PROMPT_LOGIC', WorkerPrompts().logic),
                code=os.getenv('WORKER_PROMPT_CODE', WorkerPrompts().code),
                factual=os.getenv('WORKER_PROMPT_FACTUAL', WorkerPrompts().factual),
                creative=os.getenv('WORKER_PROMPT_CREATIVE', WorkerPrompts().creative),
                analysis=os.getenv('WORKER_PROMPT_ANALYSIS', WorkerPrompts().analysis)
            ),
            max_reflection_iterations=int(os.getenv('MAX_REFLECTION_ITERATIONS', '2')),
            use_symbolic_verification=os.getenv('USE_SYMBOLIC_VERIFICATION', 'true').lower() == 'true',
            debug_verification=os.getenv('DEBUG_VERIFICATION', 'false').lower() == 'true'
        )


