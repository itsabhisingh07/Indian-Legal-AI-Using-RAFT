import streamlit as st
import os
import chromadb
from llama_cpp import Llama


st.set_page_config(page_title="Legal RAFT AI",layout="wide")

st.title(" Legal RAFT: Domain-Specific Legal AI")
st.markdown("### Powered by Llama-3, RAG, and Indian Penal Code Data")


with st.sidebar:
    st.header(" Model Parameters")
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.1)
    max_tokens = st.slider("Max Response Length", 128, 1024, 512)
    st.divider()
    st.markdown("**System Status:**")
    status_text = st.empty()
    status_text.write(" Offline")


@st.cache_resource
def load_resources():
    
    
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(SCRIPT_DIR, "..", "data", "chroma_db")
    model_path = os.path.join(SCRIPT_DIR, "..", "models", "llama-3-8b-instruct.Q4_K_M.gguf")

    
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name="ipc_data")

   
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1, 
        n_ctx=2048,
        verbose=False
    )
    
    return collection, llm


try:
    
    status_text.write(" Loading Brain... This will take a few seconds.")
    collection, llm = load_resources()
    status_text.write(" Online & Ready")
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your legal assistant. Ask me about the Indian Penal Code or BNS."}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask your legal question..."):
    
    clean_prompt = prompt.replace('"', '').replace("'", "")
    
    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    with st.chat_message("user"):
        st.markdown(clean_prompt)

    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        
    
        legal_index = {
            "Article 1": "Name and territory of the Union",
            "Article 2": "Admission or establishment of new States",
            "Article 3": "Formation of new States and alteration of areas, boundaries or names",
            "Article 4": "Laws under Articles 2 and 3 to amend First and Fourth Schedules",
            "Article 5": "Citizenship at commencement of the Constitution",
            "Article 6": "Citizenship of persons migrating from Pakistan to India",
            "Article 7": "Citizenship of persons migrating to Pakistan",
            "Article 8": "Citizenship of persons of Indian origin residing outside India",
            "Article 9": "Voluntary acquisition of foreign citizenship terminates Indian citizenship",
            "Article 10": "Continuance of citizenship rights",
            "Article 11": "Parliament to regulate citizenship by law",

            "Article 12": "Definition of State",
            "Article 13": "Laws inconsistent with Fundamental Rights are void",
            "Article 14": "Equality before law",
            "Article 15": "Prohibition of discrimination",
            "Article 16": "Equality of opportunity in public employment",
            "Article 17": "Abolition of untouchability",
            "Article 18": "Abolition of titles",
            "Article 19": "Protection of certain freedoms",
            "Article 20": "Protection in respect of conviction for offences",
            "Article 21": "Protection of life and personal liberty",
            "Article 21A": "Right to education",
            "Article 22": "Protection against arrest and detention",
            "Article 23": "Prohibition of human trafficking and forced labour",
            "Article 24": "Prohibition of child labour in hazardous employment",
            "Article 25": "Freedom of conscience and religion",
            "Article 26": "Freedom to manage religious affairs",
            "Article 27": "Freedom from taxes for promotion of a religion",
            "Article 28": "Freedom from religious instruction in certain institutions",
            "Article 29": "Protection of interests of minorities",
            "Article 30": "Right of minorities to establish educational institutions",
            "Article 31": "Right to property (repealed)",
            "Article 31A": "Saving of laws providing for acquisition of estates",
            "Article 31B": "Validation of certain Acts and Regulations",
            "Article 31C": "Saving of laws giving effect to certain Directive Principles",
            "Article 32": "Right to constitutional remedies",
            "Article 33": "Parliament may modify rights for armed forces etc.",
            "Article 34": "Restriction on rights during martial law",
            "Article 35": "Parliament to make laws for certain Fundamental Rights",

            "Article 36": "Definition of State for Directive Principles",
            "Article 37": "Directive Principles fundamental in governance but not enforceable",
            "Article 38": "State to promote welfare of people",
            "Article 39": "Principles of policy (livelihood, equal pay, etc.)",
            "Article 39A": "Equal justice and free legal aid",
            "Article 40": "Organisation of village panchayats",
            "Article 41": "Right to work, education and assistance",
            "Article 42": "Humane conditions of work and maternity relief",
            "Article 43": "Living wage for workers",
            "Article 43A": "Workers' participation in management",
            "Article 43B": "Promotion of cooperative societies",
            "Article 44": "Uniform civil code",
            "Article 45": "Early childhood care and education",
            "Article 46": "Promotion of SC/ST and weaker sections",
            "Article 47": "Duty of State to raise nutrition and public health",
            "Article 48": "Organisation of agriculture and animal husbandry",
            "Article 48A": "Protection of environment",
            "Article 49": "Protection of monuments",
            "Article 50": "Separation of judiciary from executive",
            "Article 51": "Promotion of international peace",

            "Article 51A": "Fundamental duties of citizens",

            "Article 52": "The President of India",
            "Article 53": "Executive power of the Union",
            "Article 54": "Election of President",
            "Article 55": "Manner of election of President",
            "Article 56": "Term of office of President",
            "Article 57": "Eligibility for re-election",
            "Article 58": "Qualifications for election as President",
            "Article 59": "Conditions of President's office",
            "Article 60": "Oath of President",
            "Article 61": "Impeachment of President",
            "Article 62": "Election to fill vacancy in President's office",
            "Article 63": "The Vice-President of India",
            "Article 64": "Vice-President as Chairman of Rajya Sabha",
            "Article 65": "Vice-President to act as President",
            "Article 66": "Election of Vice-President",
            "Article 67": "Term of office of Vice-President",
            "Article 68": "Election to fill vacancy of Vice-President",
            "Article 69": "Oath of Vice-President",
            "Article 70": "Discharge of President’s functions in contingencies",
            "Article 71": "Election disputes of President and Vice-President",
            "Article 72": "President’s power to grant pardons",
            "Article 73": "Extent of executive power of Union",
            "Article 74": "Council of Ministers to aid and advise President",
            "Article 75": "Provisions as to Ministers",
            "Article 76": "Attorney-General for India",
            "Article 77": "Conduct of business of Government of India",
            "Article 78": "Duties of Prime Minister",
            "Article 79": "Constitution of Parliament",
            "Article 80": "Composition of the Council of States (Rajya Sabha)",
            "Article 81": "Composition of the House of the People (Lok Sabha)",
            "Article 82": "Readjustment of seats after each census",
            "Article 83": "Duration of Houses of Parliament",
            "Article 84": "Qualifications for membership of Parliament",
            "Article 85": "Sessions, prorogation and dissolution",
            "Article 86": "Right of President to address and send messages to Houses",
            "Article 87": "Special address by the President",
            "Article 88": "Rights of Ministers and Attorney-General in Parliament",

            "Article 89": "Chairman and Deputy Chairman of the Council of States",
            "Article 90": "Vacation and resignation of Deputy Chairman",
            "Article 91": "Power of Deputy Chairman or other person to perform duties of Chairman",
            "Article 92": "Chairman not to preside while resolution for removal is under consideration",
            "Article 93": "Speaker and Deputy Speaker of the House of the People",
            "Article 94": "Vacation and resignation of Speaker and Deputy Speaker",
            "Article 95": "Power of Deputy Speaker or other person to perform duties of Speaker",
            "Article 96": "Speaker not to preside while resolution for removal is under consideration",
            "Article 97": "Salaries and allowances of Chairman, Deputy Chairman, Speaker and Deputy Speaker",

            "Article 98": "Secretariat of Parliament",
            "Article 99": "Oath or affirmation by members",
            "Article 100": "Voting in Houses and quorum",
            "Article 101": "Vacation of seats",
            "Article 102": "Disqualifications for membership",
            "Article 103": "Decision on questions as to disqualifications of members",
            "Article 104": "Penalty for sitting and voting before making oath or when disqualified",

            "Article 105": "Powers, privileges and immunities of Parliament and its members",
            "Article 106": "Salaries and allowances of members",
            "Article 107": "Provisions as to introduction and passing of Bills",
            "Article 108": "Joint sitting of both Houses",
            "Article 109": "Special procedure in respect of Money Bills",
            "Article 110": "Definition of Money Bill",
            "Article 111": "Assent to Bills",

            "Article 112": "Annual financial statement (Budget)",
            "Article 113": "Procedure in Parliament with respect to estimates",
            "Article 114": "Appropriation Bills",
            "Article 115": "Supplementary, additional or excess grants",
            "Article 116": "Votes on account, votes of credit and exceptional grants",
            "Article 117": "Special provisions as to financial Bills",

            "Article 118": "Rules of procedure",
            "Article 119": "Regulation by law of procedure in Parliament relating to financial business",
            "Article 120": "Language to be used in Parliament",
            "Article 121": "Restriction on discussion regarding conduct of judges",
            "Article 122": "Courts not to inquire into proceedings of Parliament",

            "Article 123": "Power of President to promulgate Ordinances",

            "Article 124": "Establishment and constitution of Supreme Court",
            "Article 124A": "National Judicial Appointments Commission (inserted, later struck down)",
            "Article 124B": "Functions of National Judicial Appointments Commission",
            "Article 124C": "Power of Parliament to make law regarding NJAC",
            "Article 125": "Salaries of Judges",
            "Article 126": "Appointment of acting Chief Justice",
            "Article 127": "Appointment of ad hoc Judges",
            "Article 128": "Attendance of retired Judges",
            "Article 129": "Supreme Court to be a court of record",
            "Article 130": "Seat of Supreme Court",
            "Article 131": "Original jurisdiction of Supreme Court",
            "Article 131A": "Exclusive jurisdiction of Supreme Court in constitutional validity matters (repealed)",
            "Article 132": "Appellate jurisdiction in constitutional cases",
            "Article 133": "Appellate jurisdiction in civil cases",
            "Article 134": "Appellate jurisdiction in criminal cases",
            "Article 134A": "Certificate for appeal to Supreme Court",
            "Article 135": "Jurisdiction of Federal Court vested in Supreme Court",
            "Article 136": "Special leave to appeal",
            "Article 137": "Review of judgments",
            "Article 138": "Enlargement of jurisdiction of Supreme Court",
            "Article 139": "Conferment of writ powers",
            "Article 139A": "Transfer of certain cases",
            "Article 140": "Ancillary powers of Supreme Court",
            "Article 141": "Law declared by Supreme Court binding on all courts",
            "Article 142": "Enforcement of decrees and orders of Supreme Court",
            "Article 143": "Advisory jurisdiction of Supreme Court",
            "Article 144": "Civil and judicial authorities to act in aid of Supreme Court",
            "Article 145": "Rules of Court",
            "Article 146": "Officers and servants of Supreme Court",
            "Article 147": "Interpretation",

            "Article 148": "Comptroller and Auditor-General of India",
            "Article 149": "Duties and powers of Comptroller and Auditor-General",
            "Article 150": "Form of accounts of Union and States",
            "Article 151": "Audit reports",
            "Article 152": "Definition of State (for Part VI)",
            "Article 153": "Governors of States",
            "Article 154": "Executive power of the State",
            "Article 155": "Appointment of Governor",
            "Article 156": "Term of office of Governor",
            "Article 157": "Qualifications for appointment as Governor",
            "Article 158": "Conditions of Governor’s office",
            "Article 159": "Oath or affirmation by Governor",
            "Article 160": "Discharge of functions of Governor in contingencies",
            "Article 161": "Power of Governor to grant pardons",
            "Article 162": "Extent of executive power of State",

            "Article 163": "Council of Ministers to aid and advise Governor",
            "Article 164": "Other provisions as to Ministers",
            "Article 165": "Advocate-General for the State",
            "Article 166": "Conduct of business of State Government",
            "Article 167": "Duties of Chief Minister",

            "Article 168": "Constitution of State Legislatures",
            "Article 169": "Abolition or creation of Legislative Councils",
            "Article 170": "Composition of Legislative Assemblies",
            "Article 171": "Composition of Legislative Councils",
            "Article 172": "Duration of State Legislatures",
            "Article 173": "Qualification for membership of State Legislature",
            "Article 174": "Sessions of State Legislature",
            "Article 175": "Right of Governor to address and send messages",
            "Article 176": "Special address by Governor",
            "Article 177": "Rights of Ministers and Advocate-General in Legislature",

            "Article 178": "Speaker and Deputy Speaker of Legislative Assembly",
            "Article 179": "Vacation and resignation of Speaker and Deputy Speaker",
            "Article 180": "Power of Deputy Speaker to perform duties of Speaker",
            "Article 181": "Speaker not to preside when resolution for removal is under consideration",
            "Article 182": "Chairman and Deputy Chairman of Legislative Council",
            "Article 183": "Vacation and resignation of Chairman and Deputy Chairman",
            "Article 184": "Power of Deputy Chairman to perform duties of Chairman",
            "Article 185": "Chairman not to preside when resolution for removal is under consideration",
            "Article 186": "Salaries and allowances of Speaker, Deputy Speaker, Chairman and Deputy Chairman",

            "Article 187": "Secretariat of State Legislature",
            "Article 188": "Oath or affirmation by members",
            "Article 189": "Voting in Houses and quorum",
            "Article 190": "Vacation of seats",
            "Article 191": "Disqualifications for membership",
            "Article 192": "Decision on disqualifications of members",
            "Article 193": "Penalty for sitting and voting before oath or when disqualified",

            "Article 194": "Powers, privileges and immunities of State Legislature",
            "Article 195": "Salaries and allowances of members",
            "Article 196": "Introduction and passing of Bills",
            "Article 197": "Restriction on powers of Legislative Council",
            "Article 198": "Special procedure for Money Bills",
            "Article 199": "Definition of Money Bill (State)",
            "Article 200": "Assent to Bills",
            "Article 201": "Bills reserved for consideration of President",

            "Article 202": "Annual financial statement (State Budget)",
            "Article 203": "Procedure in Legislature regarding estimates",
            "Article 204": "Appropriation Bills",
            "Article 205": "Supplementary, additional or excess grants",
            "Article 206": "Votes on account, votes of credit and exceptional grants",
            "Article 207": "Special provisions as to financial Bills",

            "Article 208": "Rules of procedure",
            "Article 209": "Regulation by law of procedure in financial business",
            "Article 210": "Language to be used in Legislature",
            "Article 211": "Restriction on discussion regarding conduct of judges",
            "Article 212": "Courts not to inquire into proceedings of Legislature",

            "Article 213": "Power of Governor to promulgate Ordinances",

            "Article 214": "High Courts for States",
            "Article 215": "High Courts to be courts of record",
            "Article 216": "Constitution of High Courts",
            "Article 217": "Appointment and conditions of office of High Court Judges",
            "Article 218": "Application of certain provisions relating to Supreme Court to High Courts",
            "Article 219": "Oath or affirmation by Judges",
            "Article 220": "Restriction on practice after being Judge",
            "Article 221": "Salaries of Judges",
            "Article 222": "Transfer of Judges",
            "Article 223": "Appointment of acting Chief Justice",
            "Article 224": "Appointment of additional and acting Judges",
            "Article 224A": "Appointment of retired Judges at sittings of High Courts",
            "Article 225": "Jurisdiction of existing High Courts",
            "Article 226": "Power of High Courts to issue writs",
            "Article 227": "Power of superintendence over all courts",
            "Article 228": "Transfer of certain cases to High Court",
            "Article 228A": "Special provisions for constitutional validity (repealed)",
            "Article 229": "Officers and servants of High Courts",
            "Article 230": "Extension of jurisdiction of High Courts to Union territories",
            "Article 231": "Establishment of common High Court for two or more States",

            "Article 232": "Omitted",
            "Article 233": "Appointment of district judges",
            "Article 233A": "Validation of appointments of district judges",
            "Article 234": "Recruitment of persons other than district judges to judicial service",
            "Article 235": "Control over subordinate courts",
            "Article 236": "Interpretation",
            "Article 237": "Application of provisions to certain magistrates",
            "Article 238": "Application of provisions relating to States to certain Part B States (repealed)",

            "Article 239": "Administration of Union Territories",
            "Article 239A": "Creation of local Legislatures or Council of Ministers for certain Union Territories",
            "Article 239AA": "Special provisions for Delhi",
            "Article 239AB": "Provision in case of failure of constitutional machinery in Delhi",
            "Article 239B": "Power of Administrator to promulgate Ordinances",
            "Article 240": "Power of President to make regulations for certain Union Territories",
            "Article 241": "High Courts for Union Territories",
            "Article 242": "Omitted",

            "Article 243": "Definitions (Panchayats)",
            "Article 243A": "Gram Sabha",
            "Article 243B": "Constitution of Panchayats",
            "Article 243C": "Composition of Panchayats",
            "Article 243D": "Reservation of seats",
            "Article 243E": "Duration of Panchayats",
            "Article 243F": "Disqualifications for membership",
            "Article 243G": "Powers, authority and responsibilities of Panchayats",
            "Article 243H": "Powers to impose taxes by Panchayats",
            "Article 243I": "Finance Commission for Panchayats",
            "Article 243J": "Audit of accounts of Panchayats",
            "Article 243K": "Elections to Panchayats",
            "Article 243L": "Application to Union Territories",
            "Article 243M": "Part not to apply to certain areas",
            "Article 243N": "Continuance of existing laws",
            "Article 243O": "Bar to interference by courts in electoral matters",

            "Article 243P": "Definitions (Municipalities)",
            "Article 243Q": "Constitution of Municipalities",
            "Article 243R": "Composition of Municipalities",
            "Article 243S": "Constitution and composition of Wards Committees",
            "Article 243T": "Reservation of seats in Municipalities",
            "Article 243U": "Duration of Municipalities",
            "Article 243V": "Disqualifications for membership",
            "Article 243W": "Powers, authority and responsibilities of Municipalities",
            "Article 243X": "Power to impose taxes by Municipalities",
            "Article 243Y": "Finance Commission",
            "Article 243Z": "Audit of accounts of Municipalities",
            "Article 243ZA": "Elections to Municipalities",
            "Article 243ZB": "Application to Union Territories",
            "Article 243ZC": "Part not to apply to certain areas",
            "Article 243ZD": "Committee for district planning",
            "Article 243ZE": "Committee for metropolitan planning",
            "Article 243ZF": "Continuance of existing laws",
            "Article 243ZG": "Bar to interference by courts in electoral matters",

            "Article 244": "Administration of Scheduled Areas and Tribal Areas",
            "Article 244A": "Formation of autonomous State in Assam",
            "Article 245": "Extent of laws made by Parliament and State Legislatures",
            "Article 246": "Subject-matter of laws made by Parliament and State Legislatures",
            "Article 246A": "Special provision for Goods and Services Tax",
            "Article 247": "Power of Parliament to provide for additional courts",
            "Article 248": "Residuary powers of legislation",
            "Article 249": "Power of Parliament to legislate on State List in national interest",
            "Article 250": "Power of Parliament to legislate during Emergency",
            "Article 251": "Inconsistency between laws made by Parliament and State Legislatures",
            "Article 252": "Power of Parliament to legislate for two or more States by consent",
            "Article 253": "Legislation for implementing international agreements",
            "Article 254": "Inconsistency between laws made by Parliament and State laws",
            "Article 255": "Requirements as to recommendations and previous sanctions",

            "Article 256": "Obligation of States and Union",
            "Article 257": "Control of the Union over States in certain cases",
            "Article 258": "Power of Union to confer powers on States",
            "Article 258A": "Power of States to entrust functions to Union",
            "Article 259": "Omitted",
            "Article 260": "Jurisdiction of Union in relation to territories outside India",
            "Article 261": "Public acts, records and judicial proceedings",
            "Article 262": "Adjudication of disputes relating to waters of inter-State rivers",
            "Article 263": "Provisions with respect to Inter-State Council",

            "Article 264": "Interpretation (Finance)",
            "Article 265": "Taxes not to be imposed except by authority of law",
            "Article 266": "Consolidated Funds and Public Accounts",
            "Article 267": "Contingency Fund",
            "Article 268": "Duties levied by Union but collected by States",
            "Article 269": "Taxes levied and collected by Union but assigned to States",
            "Article 269A": "Levy and collection of GST in course of inter-State trade",
            "Article 270": "Taxes levied and distributed between Union and States",
            "Article 271": "Surcharge on certain duties and taxes",
            "Article 272": "Omitted",
            "Article 273": "Grants in lieu of export duty",
            "Article 274": "Prior recommendation of President required for certain Bills",
            "Article 275": "Grants from the Union to certain States",
            "Article 276": "Taxes on professions, trades and employments",
            "Article 277": "Savings",
            "Article 278": "Omitted",
            "Article 279": "Calculation of net proceeds",
            "Article 279A": "Goods and Services Tax Council",
            "Article 280": "Finance Commission",
            "Article 281": "Recommendations of Finance Commission",
            "Article 282": "Expenditure defrayable by Union or State",
            "Article 283": "Custody of Consolidated Funds",
            "Article 284": "Custody of public money",
            "Article 285": "Exemption of Union property from State taxation",
            "Article 286": "Restrictions on imposition of tax on sale or purchase of goods",
            "Article 287": "Exemption from electricity taxes in certain cases",
            "Article 288": "Exemption from taxation by States in respect of water or electricity",
            "Article 289": "Exemption of State property from Union taxation",
            "Article 290": "Adjustment in respect of certain expenses and pensions",
            "Article 290A": "Annual payment to certain Devaswom Funds",
            "Article 291": "Omitted",
            "Article 292": "Borrowing by the Government of India",
            "Article 293": "Borrowing by States",

            "Article 294": "Succession to property, assets, rights and liabilities",
            "Article 295": "Succession to property, assets, rights and liabilities in other cases",
            "Article 296": "Property accruing by escheat or lapse",
            "Article 297": "Things of value within territorial waters or continental shelf",
            "Article 298": "Power to carry on trade",
            "Article 299": "Contracts",
            "Article 300": "Suits and proceedings",
            "Article 300A": "Right to property",

            "Article 301": "Freedom of trade, commerce and intercourse",
            "Article 302": "Power of Parliament to impose restrictions",
            "Article 303": "Restrictions on legislative powers of Union and States",
            "Article 304": "Restrictions on trade by States",
            "Article 305": "Saving of existing laws",
            "Article 306": "Omitted",
            "Article 307": "Authority for carrying out trade provisions",

            "Article 308": "Interpretation (Services)",
            "Article 309": "Recruitment and conditions of service",
            "Article 310": "Tenure of office of persons serving the Union or State",
            "Article 311": "Dismissal, removal or reduction in rank of civil servants",
            "Article 312": "All-India Services",
            "Article 312A": "Power of Parliament to vary or revoke conditions of service",
            "Article 313": "Transitional provisions",
            "Article 314": "Omitted",

            "Article 315": "Public Service Commissions for Union and States",
            "Article 316": "Appointment and term of office of members",
            "Article 317": "Removal and suspension of members",
            "Article 318": "Power to make regulations",
            "Article 319": "Prohibition of holding office after ceasing to be member",
            "Article 320": "Functions of Public Service Commissions",
            "Article 321": "Extension of functions",
            "Article 322": "Expenses of Public Service Commissions",
            "Article 323": "Reports of Public Service Commissions",

            "Article 323A": "Administrative Tribunals",
            "Article 323B": "Tribunals for other matters",

            "Article 324": "Superintendence of elections vested in Election Commission",
            "Article 325": "No person to be ineligible for inclusion in electoral rolls",
            "Article 326": "Elections based on adult suffrage",
            "Article 327": "Power of Parliament to make provision regarding elections",
            "Article 328": "Power of State Legislature to make provision regarding elections",
            "Article 329": "Bar to interference by courts in electoral matters",
            "Article 329A": "Special provision relating to election of Prime Minister and Speaker (repealed)",

            "Article 330": "Reservation of seats for SC/ST in Lok Sabha",
            "Article 331": "Representation of Anglo-Indian community in Lok Sabha (repealed)",
            "Article 332": "Reservation of seats for SC/ST in State Assemblies",
            "Article 333": "Representation of Anglo-Indian community in State Assemblies (repealed)",
            "Article 334": "Reservation of seats and special representation to cease after certain period",
            "Article 335": "Claims of SC/ST in services",
            "Article 336": "Special provision for Anglo-Indian community (certain services)",
            "Article 337": "Special provision for Anglo-Indian educational institutions",
            "Article 338": "National Commission for Scheduled Castes",
            "Article 338A": "National Commission for Scheduled Tribes",
            "Article 338B": "National Commission for Backward Classes",
            "Article 339": "Control of Union over administration of Scheduled Areas",
            "Article 340": "Appointment of Commission to investigate backward classes",
            "Article 341": "Scheduled Castes",
            "Article 342": "Scheduled Tribes",
            "Article 342A": "Socially and Educationally Backward Classes",

            "Article 343": "Official language of the Union",
            "Article 344": "Commission and Committee on official language",
            "Article 345": "Official language of a State",
            "Article 346": "Official language for communication between States",
            "Article 347": "Special provision relating to language spoken by a section",
            "Article 348": "Language of Supreme Court and High Courts",
            "Article 349": "Special procedure for enactment of language laws",
            "Article 350": "Language to be used in representations for redress",
            "Article 350A": "Facilities for instruction in mother tongue",
            "Article 350B": "Special Officer for linguistic minorities",
            "Article 351": "Directive for development of Hindi language",

            "Article 352": "Proclamation of Emergency",
            "Article 353": "Effect of Proclamation of Emergency",
            "Article 354": "Application of provisions relating to distribution of revenues during Emergency",
            "Article 355": "Duty of Union to protect States",
            "Article 356": "Failure of constitutional machinery in States",
            "Article 357": "Exercise of legislative powers during President's Rule",
            "Article 358": "Suspension of provisions of Article 19 during Emergency",
            "Article 359": "Suspension of enforcement of Fundamental Rights",
            "Article 360": "Financial Emergency",

            "Article 361": "Protection of President and Governors",
            "Article 361A": "Protection of publication of proceedings",
            "Article 361B": "Disqualification under anti-defection law",
            "Article 362": "Omitted",
            "Article 363": "Bar to interference by courts in disputes arising out of treaties",
            "Article 363A": "Recognition granted to Rulers of Indian States to cease",
            "Article 364": "Special provisions relating to major ports and aerodromes",
            "Article 365": "Effect of failure to comply with Union directions",
            "Article 366": "Definitions",
            "Article 367": "Interpretation",

            "Article 368": "Power of Parliament to amend the Constitution",

            "Article 369": "Temporary power of Parliament to legislate on State List matters",
            "Article 370": "Temporary provisions with respect to Jammu and Kashmir",
            "Article 371": "Special provisions for certain States",
            "Article 371A": "Special provision for Nagaland",
            "Article 371B": "Special provision for Assam",
            "Article 371C": "Special provision for Manipur",
            "Article 371D": "Special provision for Andhra Pradesh",
            "Article 371E": "Establishment of Central University in Andhra Pradesh",
            "Article 371F": "Special provisions for Sikkim",
            "Article 371G": "Special provision for Mizoram",
            "Article 371H": "Special provision for Arunachal Pradesh",
            "Article 371I": "Special provision for Goa",
            "Article 371J": "Special provision for Karnataka",

            "Article 372": "Continuance of existing laws",
            "Article 372A": "Power of President to adapt laws",
            "Article 373": "Power of President in preventive detention matters",
            "Article 374": "Provisions relating to Federal Court and existing courts",
            "Article 375": "Courts and authorities to continue",
            "Article 376": "Provisions relating to Judges of High Courts",
            "Article 377": "Provisions relating to Comptroller and Auditor-General",
            "Article 378": "Provisions relating to Public Service Commissions",
            "Article 378A": "Special provisions as to duration of Andhra Pradesh Legislative Assembly (historical)",
            "Article 379": "Omitted",
            "Article 380": "Omitted",
            "Article 381": "Omitted",
            "Article 382": "Omitted",
            "Article 383": "Omitted",
            "Article 384": "Omitted",
            "Article 385": "Omitted",
            "Article 386": "Omitted",
            "Article 387": "Omitted",
            "Article 388": "Omitted",
            "Article 389": "Omitted",
            "Article 390": "Omitted",
            "Article 391": "Omitted",
            "Article 392": "Power of President to remove difficulties",
            "Article 393": "Short title",
            "Article 394": "Commencement",
            "Article 394A": "Authoritative text in Hindi",
            "Article 395": "Repeals",
            "section 379": "punishment for theft",
            "section 105": "culpable homicide not amounting to murder",
            "section 103": "punishment for murder",
            "murder": "103.Whoever commits murder shall be punished with death or imprisonment" 

        }
        
        search_query = clean_prompt
        for keyword, meaning in legal_index.items():
            if keyword.lower() in clean_prompt.lower():
                search_query = f"{clean_prompt} {meaning}"
                st.caption(f" *AI expanded query to search for concept: '{meaning}'*")
                break
        
        results = collection.query(
            query_texts=[search_query], 
            n_results=5,
            include=["documents", "metadatas", "distances"] 
        )
        
        
        valid_docs = []
        valid_sources = []
        
        st.sidebar.markdown("###  AI Debug Mode (Distance Scores)") 
        
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            st.sidebar.write(f"Distance: `{dist:.4f}`") 
            
            
            if dist < 1:  
                valid_docs.append(doc)
                valid_sources.append(meta['source'])
        
      
        if len(valid_docs) == 0:
            safe_response = "I cannot find the relevant information in the provided legal text."
            message_placeholder.markdown(safe_response)
            st.session_state.messages.append({"role": "assistant", "content": safe_response})
            st.stop() 
            
        retrieved_text = "\n\n".join(valid_docs)
        sources = list(set(valid_sources))
        
        
        system_instruction = (
            "You are a strict Indian legal assistant. "
            "You must answer the user's question USING ONLY the provided Legal Context. "
            "CRITICAL RULES:\n"
            "1. If the context describes multiple variations of a crime, list them separately.\n"
            "2. NEVER combine different punishments into one single sentence.\n"
            "3. NO DISCLAIMERS. Absolutely do not say 'I cannot provide legal advice'. Start your answer directly with the facts.\n"
            "4. ANTI-HALLUCINATION: If the exact answer is not in the context, you must say EXACTLY: 'I cannot find the relevant information in the provided legal text.' Do NOT guess.\n\n"
            f"Legal Context:\n{retrieved_text}"
        )
        
        prefill = "Based directly on the provided text, "
        full_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{prefill}"

      
        full_response = prefill 
        
        stream = llm.create_completion(
            full_prompt,
            max_tokens=max_tokens,
            temperature=0.1, 
            top_p=0.9,       
            stream=True
        )
        
        for chunk in stream:
            if "text" in chunk["choices"][0]:
                text_chunk = chunk["choices"][0]["text"]
                full_response += text_chunk
                message_placeholder.markdown(full_response + "▌")
        
        
        message_placeholder.markdown(full_response)
        st.caption(f" Sources Analyzed: {', '.join(sources)}")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})