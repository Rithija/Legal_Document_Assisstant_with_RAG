import os
import re
import pickle
from pypdf import PdfReader
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import re
from datetime import datetime

def extract_metadata(page_one_text):
    """
    Extracts metadata from the first page of a Supreme Court judgment.
    This updated version is more robust to handle different document formats.
    """
    metadata = {
        "case_id": "Not Found",
        "case_name": "Not Found",
        "citation": "Not Found",
        "court_name": "IN THE SUPREME COURT OF INDIA",
        "judges": [],
        "decision_date": "Not Found"
    }

    # 1. Decision Date 
    date_match = re.search(r"on (\d{1,2} \w+, \d{4})", page_one_text, re.IGNORECASE)
    if date_match:
        try:
            date_obj = datetime.strptime(date_match.group(1), "%d %B, %Y")
            metadata["decision_date"] = date_obj.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 2. Case Name 
    try:
        first_line = [line for line in page_one_text.split('\n') if line.strip()][0]
        delimiter = " on "
        
        if delimiter in first_line:
            case_name_raw = first_line.split(delimiter)[0]
            metadata["case_name"] = " ".join(case_name_raw.strip().split())
        else:
            metadata["case_name"] = first_line.strip()
            
    except IndexError:
        pass

    # 3. Judges 
    judges_found = []
    bench_match = re.search(r"Bench: (.*)", page_one_text)
    if bench_match:
        judges = [j.strip() for j in bench_match.group(1).split(',')]
        judges_found.extend(judges)

    author_match = re.search(r"Author: (.*)", page_one_text)
    if author_match:
        judges_found.append(author_match.group(1).strip())

    judge_matches = re.findall(r"^(?!.*\bJUDGMENT\b)([A-Z.\s]+, J\.)", page_one_text, re.MULTILINE | re.IGNORECASE)
    if judge_matches:
        judges_found.extend(judge_matches)
    
    if judges_found:
        cleaned_judges = set()
        for entry in judges_found:
            for potential_judge in entry.split('\n'):
               
                clean_name = re.sub(r'[,.\s]*J\.?$', '', potential_judge, flags=re.IGNORECASE).strip(' ,.')

                if clean_name and 'J U D G M E N T' not in clean_name.upper():
                    standardized_name = " ".join(clean_name.title().split())
                    cleaned_judges.add(standardized_name)
        
        metadata["judges"] = sorted(list(cleaned_judges))


    id_match = re.search(r"APPEAL NO(?:S)?\.\s+([\d-]+)", page_one_text, re.IGNORECASE)
    if id_match:
        case_id = id_match.group(1).strip()
        metadata["case_id"] = f"SC_{case_id}"

    citations_match = re.search(r"Equivalent citations: (.*?)(?=\nAuthor:|\nBench:)", page_one_text, re.DOTALL)
    if citations_match:
        metadata["citation"] = " ".join(citations_match.group(1).strip().split())
    elif metadata["case_id"] != "Not Found":
        metadata["citation"] = metadata["case_id"]

    return metadata

def split_into_semantic_sections(full_text):
    """
    Splits the full text of a judgment into a dictionary of semantic sections
    based on a predefined list of common legal document headings.
    """
    section_keywords = [
        "BRIEF FACTS:",
        "SUBMISSIONS BY THE APPELLANT",
        "SUBMISSIONS BY THE RESPONDENT",
        "ANALYSIS, REASONING & CONCLUSION",
        "ANALYSIS AND REASONING",
        "ANALYSIS",
        "REASONING",
        "CONCLUSION",
        "JUDGMENT"
    ]
    
    normalized_text = " ".join(full_text.split())
    
    found_sections = []
    for keyword in section_keywords:
        try:
            for match in re.finditer(re.escape(keyword), normalized_text, re.IGNORECASE):
                found_sections.append((match.start(), keyword))
        except re.error:
            continue

    found_sections.sort()
    
    sections_dict = {}
    if not found_sections:
        sections_dict["full_text"] = normalized_text
        return sections_dict

    for i, (start_pos, keyword) in enumerate(found_sections):
        end_pos = None
        if i + 1 < len(found_sections):
            end_pos = found_sections[i+1][0]
        
        section_text = normalized_text[start_pos:end_pos]
        
        clean_key = keyword.replace(":", "").replace("-", " ").strip()
        sections_dict[clean_key] = section_text
        
    return sections_dict

def process_and_chunk_files():
    """
    Main function to orchestrate the processing of all PDF files.
    It reads, extracts metadata, splits into sections, chunks the sections,
    and saves the final data.
    """
    data_directory = "supreme_court_judgments/"
    all_chunks = []

    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    

    pdf_files = []
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    print(f"Found {len(pdf_files)} PDF files to process.") 

    

    for full_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            filename = os.path.basename(full_path) 
            
            # --- 1. Read PDF & Extract Metadata ---
            reader = PdfReader(full_path)
            page_one_text = reader.pages[0].extract_text() or ""
            metadata = extract_metadata(page_one_text)
            metadata['source_file'] = filename
            
            full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

            # --- 2. Split into Logical Sections ---
            sections = split_into_semantic_sections(full_text)
            
            # --- 3. Chunk Within Sections & Store ---
            for section_title, section_text in sections.items():
                sub_chunks = chunk_splitter.split_text(section_text)
                
                for chunk_text in sub_chunks:
                    chunk_metadata = metadata.copy()
                    chunk_metadata['section'] = section_title
                    
                    doc = Document(page_content=chunk_text, metadata=chunk_metadata)
                    all_chunks.append(doc)

        except Exception as e:
            print(f"\n[!] Error processing {filename}: {e}")
            continue 
    # --- 4. Save the Final Processed Data ---
    output_filename = "processed_chunks.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(all_chunks, f)
        
    print(f"\n✅ Processing complete. Total chunks created: {len(all_chunks)}")
    print(f"All processed chunks have been saved to '{output_filename}'")


if __name__ == "__main__":
    process_and_chunk_files()