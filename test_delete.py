import json

def build_sidebar_html(histories, active_sid=""):
    return str(list(histories.keys()))

def delete_chat(target_payload, current_sid, all_histories):
    target_sid = target_payload.split("||")[0] if target_payload else ""
    histories = json.loads(all_histories) if all_histories else {}

    print(f"To Delete: {target_sid}")
    print(f"Before: {list(histories.keys())}")

    if target_sid in histories:
        del histories[target_sid]

    print(f"After: {list(histories.keys())}")

    new_histories_json = json.dumps(histories)

    if current_sid == target_sid:
        if histories:
            new_sid = list(histories.keys())[-1]
            msgs = histories[new_sid]["messages"] or ["WELCOME"]
        else:
            new_sid = ""
            msgs = ["WELCOME"]
    else:
        new_sid = current_sid
        msgs = histories.get(current_sid, {}).get("messages") or ["WELCOME"]

    return msgs, new_sid, new_histories_json, build_sidebar_html(histories, new_sid)


hist = {"Chat 1": {"messages": []}, "Chat 2": {"messages": []}, "Chat 3": {"messages": []}}

# Test 1: Delete active chat (Chat 3)
msgs, new_sid, new_hist, sidebar = delete_chat("Chat 3||123", "Chat 3", json.dumps(hist))
print(f"Result 1: msgs={msgs}, new_sid={new_sid}, sidebar={sidebar}\n")

# Test 2: Delete background chat (Chat 1)
msgs, new_sid, new_hist, sidebar = delete_chat("Chat 1||123", "Chat 2", json.dumps(hist))
print(f"Result 2: msgs={msgs}, new_sid={new_sid}, sidebar={sidebar}\n")

