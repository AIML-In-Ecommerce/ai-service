from llama_index.core.tools import FunctionTool
from app.services.genai_service import generateChart

chart_engine = FunctionTool.from_defaults(
    fn = generateChart, name = "gen_chart",
    description=""
)
