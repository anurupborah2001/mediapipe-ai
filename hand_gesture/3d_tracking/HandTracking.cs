using System.Globalization;
using UnityEngine;

public class HandTracking : MonoBehaviour
{
    public UDPReceive udpReceive;
    public GameObject[] handObjects;

    void Update()
    {
        string data = udpReceive.data;

        if (string.IsNullOrWhiteSpace(data))
            return;


        data = data.Trim();

        // Remove JSON array brackets: [1,2,3,...]
        if (data.StartsWith("["))
            data = data.Substring(1);

        if (data.EndsWith("]"))
            data = data.Substring(0, data.Length - 1);

        string[] handData = data.Split(',');

        // 21 landmarks * x,y,z = 63 values
        if (handData.Length < 63)
        {
            Debug.LogWarning($"Invalid landmark data. Expected 63 values, got {handData.Length}");
            return;
        }

        for (int i = 0; i < 21; i++)
        {
            if (
                float.TryParse(handData[i * 3], NumberStyles.Float, CultureInfo.InvariantCulture, out float rawX) &&
                float.TryParse(handData[i * 3 + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out float rawY) &&
                float.TryParse(handData[i * 3 + 2], NumberStyles.Float, CultureInfo.InvariantCulture, out float rawZ)
            )
            {
                float x = 8 - rawX / 100f;
                float y = rawY / 100f;
                float z = rawZ / 100f;

                handObjects[i].transform.localPosition = new Vector3(x, y, z);
            }
        }
    }
}