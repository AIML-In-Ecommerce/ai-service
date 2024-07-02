from llama_index.core.tools import FunctionTool
from app.services.genai_service import generateChart

chart_engine = FunctionTool.from_defaults(
    fn = generateChart, name = "gen_chart",
    description="The tool performs data analysis and visualization by drawing charts. The parameter data is a string representing an object data for drawing that was obtained in the conversation"
)