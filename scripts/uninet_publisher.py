import os
import xmlrpc.client
import requests
from dotenv import load_dotenv

load_dotenv()

DOKUWIKI_URL = os.getenv("DOKUWIKI_URL")
DOKUWIKI_USER = os.getenv("DOKUWIKI_USER")
DOKUWIKI_PASS = os.getenv("DOKUWIKI_PASS")

NEXTCLOUD_URL = os.getenv("NEXTCLOUD_URL")
NEXTCLOUD_USER = os.getenv("NEXTCLOUD_USER")
NEXTCLOUD_PASS = os.getenv("NEXTCLOUD_PASS")

def get_dokuwiki_server():
    if not DOKUWIKI_URL:
        return None
    base_url = DOKUWIKI_URL.rstrip('/')
    if not base_url.endswith('/dokuwiki'):
        base_url += '/dokuwiki'
    xmlrpc_url = f"{base_url}/lib/exe/xmlrpc.php"
    if "://" in xmlrpc_url:
        protocol, rest = xmlrpc_url.split("://", 1)
        auth_url = f"{protocol}://{DOKUWIKI_USER}:{DOKUWIKI_PASS}@{rest}"
    else:
        auth_url = f"http://{DOKUWIKI_USER}:{DOKUWIKI_PASS}@{xmlrpc_url}"
    return xmlrpc.client.ServerProxy(auth_url), base_url

def list_dokuwiki_pages(namespace=""):
    try:
        server_info = get_dokuwiki_server()
        if not server_info: return []
        server, _ = server_info
        pages = server.dokuwiki.getPagelist(namespace, {"depth": 0})
        return [p.get('id') for p in pages if 'id' in p]
    except Exception as e:
        print(f"Error listing dokuwiki pages: {e}")
        return []

def read_dokuwiki_page(page_id):
    try:
        server_info = get_dokuwiki_server()
        if not server_info: return None
        server, _ = server_info
        return server.wiki.getPage(page_id)
    except Exception as e:
        print(f"Error reading dokuwiki page: {e}")
        return None

def publish_to_dokuwiki(title, content, namespace="ulisse"):
    try:
        server_info = get_dokuwiki_server()
        if not server_info:
            return False, "DOKUWIKI_URL non configurato in .env"
        
        server_auth, base_url = server_info

        
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip().replace(" ", "_").lower()
        page_id = f"{namespace}:{safe_title}"
        
        full_content = f"{content}\n\n---\nPubblicato da Ulisse Brain"
        
        # wiki.putPage
        success = server_auth.wiki.putPage(page_id, full_content, {"sum": "Automated publish by Ulisse Brain"})
        
        if success:
            page_url = f"{base_url}/doku.php?id={page_id}"
            return True, page_url
        else:
            return False, "DokuWiki returned false"
    except Exception as e:
        return False, str(e).replace('<', '[').replace('>', ']')

def publish_to_nextcloud(filename, content):
    try:
        if not NEXTCLOUD_URL:
            return False, "NEXTCLOUD_URL non configurato in .env"
            
        base_url = NEXTCLOUD_URL.rstrip('/')
        webdav_base = f"{base_url}/remote.php/webdav"
        
        auth = (NEXTCLOUD_USER, NEXTCLOUD_PASS)
        
        folders = ["Ulisse", "Ulisse/ricerche"]
        for f in folders:
            folder_url = f"{webdav_base}/{f}"
            r = requests.request("PROPFIND", folder_url, auth=auth)
            if r.status_code == 404:
                mk = requests.request("MKCOL", folder_url, auth=auth)
                if mk.status_code not in (201, 204, 200, 405):
                    return False, f"Failed to create folder {f}: {mk.status_code}"
                    
        safe_filename = "".join(c if c.isalnum() or c in " -_" else "_" for c in filename).strip().replace(" ", "_").lower()
        if not safe_filename.endswith(".md"):
            safe_filename += ".md"
            
        file_url = f"{webdav_base}/Ulisse/ricerche/{safe_filename}"
        
        put_r = requests.put(file_url, data=content.encode('utf-8'), auth=auth)
        
        if put_r.status_code in (200, 201, 204):
            return True, file_url
        else:
            return False, f"Nextcloud returned {put_r.status_code}: {put_r.text.replace('<', '[').replace('>', ']')}"
            
    except Exception as e:
        return False, str(e).replace('<', '[').replace('>', ']')

def read_ulisse_nextcloud_files():
    try:
        if not NEXTCLOUD_URL: return []
        base_url = NEXTCLOUD_URL.rstrip('/')
        webdav_base = f"{base_url}/remote.php/webdav"
        auth = (NEXTCLOUD_USER, NEXTCLOUD_PASS)
        folder_url = f"{webdav_base}/Ulisse/ricerche"
        
        r = requests.request("PROPFIND", folder_url, auth=auth, headers={"Depth": "1"})
        if r.status_code not in (207, 200):
            return []
            
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        filenames = []
        for href in root.iter('{DAV:}href'):
            if href.text:
                path = href.text
                if path.endswith('.md'):
                    name = path.split('/')[-1]
                    filenames.append(name)
        return filenames
    except Exception as e:
        print(f"Error reading nextcloud files: {e}")
        return []
